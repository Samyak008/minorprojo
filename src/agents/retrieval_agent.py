import os
import pickle
import numpy as np
from agents.base_agent import BaseAgent
from utils.text_embeddings import TextEmbeddings
from models.paper import Paper
import json
from rank_bm25 import BM25Okapi
import logging
from utils.arxiv_fetcher import ArxivFetcher
from utils.semantic_scholar_fetcher import SemanticScholarFetcher
from utils.core_fetcher import CoreFetcher
from utils.crossref_fetcher import CrossrefFetcher  
from utils.openalex_fetcher import OpenAlexFetcher
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import FAISS but handle failure gracefully
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library loaded successfully")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS library not available. Will fall back to BM25 search only.")

class RetrievalAgent(BaseAgent):
    def __init__(self, index_path):
        """
        Initialize the RetrievalAgent with a path to store/load the paper index.
        
        Args:
            index_path (str): Path to the paper index directory
        """
        self.index_path = index_path
        self.text_embedder = TextEmbeddings()
        self.papers = []
        self.paper_texts = []
        self.faiss_index = None
        self.bm25 = None
        
        # Initialize external API fetchers
        self.arxiv_fetcher = ArxivFetcher(max_results=10)
        self.semantic_scholar_fetcher = SemanticScholarFetcher()
        
        # Replace IEEE and Springer with free alternatives
        self.core_fetcher = CoreFetcher(max_results=10)
        self.crossref_fetcher = CrossrefFetcher(max_results=10)  
        self.openalex_fetcher = OpenAlexFetcher(max_results=10)
        
        self.initialize()
        
    def initialize(self):
        """
        Initialize the agent by loading the paper index or creating a new one if it doesn't exist.
        """
        logger.info(f"Initializing RetrievalAgent with index path: {self.index_path}")
        
        # Create index directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Try to load existing index
        try:
            self._load_index()
            logger.info("Successfully loaded existing paper index")
        except (FileNotFoundError, EOFError):
            logger.info("No existing index found. Creating a new one.")
            # Initialize empty indexes
            self.papers = []
            self.paper_texts = []
            self.faiss_index = None
            self.bm25 = None
            self._create_sample_data()  # For testing purposes
            self._save_index()
    
    def _create_sample_data(self):
        """Create some sample papers for testing"""
        sample_papers = [
            Paper(
                title="Machine Learning for Healthcare", 
                authors=["Alice Smith", "Bob Johnson"],
                abstract="This paper explores applications of machine learning in healthcare, focusing on predictive diagnostics.",
                publication_year="2022"
            ),
            Paper(
                title="Deep Learning Approaches to Natural Language Processing", 
                authors=["Charlie Brown", "Diana Prince"],
                abstract="We survey recent advances in deep learning models for NLP tasks such as translation and summarization.",
                publication_year="2021"
            ),
            Paper(
                title="Reinforcement Learning in Robotics", 
                authors=["Eve Williams", "Frank Miller"],
                abstract="This study demonstrates how reinforcement learning can be applied to improve robotic control systems.",
                publication_year="2023"
            )
        ]
        
        self.papers = sample_papers
        self.paper_texts = [f"{p.title} {p.abstract}" for p in sample_papers]
        
        # Create BM25 index
        tokenized_texts = [text.lower().split() for text in self.paper_texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Create FAISS index for dense retrieval
        embeddings = self.text_embedder.generate_embeddings(self.paper_texts)
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        
        # Initialize FAISS index - using L2 distance
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
    
    def _load_index(self):
        """Load the paper index from disk"""
        if not os.path.exists(f"{self.index_path}_data.pkl"):
            raise FileNotFoundError("Index files not found")
        
        with open(f"{self.index_path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.papers = data['papers']
            self.paper_texts = data['paper_texts']
        
        with open(f"{self.index_path}_bm25.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        
        self.faiss_index = faiss.read_index(f"{self.index_path}_faiss.index")
    
    def _save_index(self):
        """Save the paper index to disk"""
        # Save papers and texts
        with open(f"{self.index_path}_data.pkl", 'wb') as f:
            pickle.dump({
                'papers': self.papers,
                'paper_texts': self.paper_texts
            }, f)
        
        # Save BM25 index
        if self.bm25 is not None:
            with open(f"{self.index_path}_bm25.pkl", 'wb') as f:
                pickle.dump(self.bm25, f)
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, f"{self.index_path}_faiss.index")
    
    def search_external(self, query):
        """
        Search for papers using external APIs with improved error handling and source-specific optimizations
        
        Args:
            query (str): The search query
            
        Returns:
            list: List of Paper objects from external sources
        """
        if query is None or not query.strip():
            return []
            
        logger.info(f"Searching external sources for: {query}")
        
        # Source-specific query optimizations
        openalex_query = query
        crossref_query = query
        
        # Crossref works better with more specific terms
        if len(query.split()) <= 2:
            crossref_query = f"{query} research paper academic"
            
        # OpenAlex works better with field-specific searches for short queries
        if len(query.split()) <= 2:
            openalex_query = f"{query} title:{query} abstract:{query}"
        
        # Concurrent API fetching to improve performance
        results = {}
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all search tasks
            future_to_source = {
                executor.submit(self.arxiv_fetcher.search_papers, query): "arxiv",
                executor.submit(self.semantic_scholar_fetcher.search_papers, query): "semantic_scholar",
                executor.submit(self.core_fetcher.search_papers, query): "core",
                executor.submit(self.crossref_fetcher.search_papers, crossref_query): "crossref",
                executor.submit(self.openalex_fetcher.search_papers, openalex_query): "openalex"
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    results[source] = future.result()
                    logger.info(f"Found {len(results[source])} papers from {source}")
                except Exception as e:
                    logger.error(f"Error with {source}: {str(e)}")
                    results[source] = []
        
        # Define minimum papers per source and maximum total papers
        min_papers_per_source = 2  # At least 2 papers from each source if available
        max_total_papers = 30
        
        # Balance results from different sources
        balanced_results = []
        
        # First, ensure we have minimum representation from each source
        for source, papers in results.items():
            if papers:
                # Take at least min_papers_per_source from each source (or all if fewer available)
                balanced_results.extend(papers[:min_papers_per_source])
        
        # Track which papers we've already included (to avoid duplicates)
        included_papers = set(self._normalize_title(paper.title) for paper in balanced_results)
        
        # Fill remaining slots with papers from any source, maintaining diversity
        remaining_slots = max_total_papers - len(balanced_results)
        if remaining_slots > 0:
            # Create a round-robin selection from each source
            source_papers = {source: papers[min_papers_per_source:] for source, papers in results.items()}
            
            # Keep going until we fill all slots or run out of papers
            while remaining_slots > 0:
                added_paper = False
                
                # Try to add one paper from each source in turn
                for source in list(source_papers.keys()):
                    if not source_papers[source]:
                        # No more papers for this source
                        source_papers.pop(source)
                        continue
                        
                    # Get next paper from this source
                    next_paper = source_papers[source].pop(0)
                    
                    # Check if this is a duplicate
                    title_key = self._normalize_title(next_paper.title)
                    if title_key not in included_papers:
                        balanced_results.append(next_paper)
                        included_papers.add(title_key)
                        remaining_slots -= 1
                        added_paper = True
                        
                    if remaining_slots <= 0:
                        break
                
                # If we couldn't add any papers in this round, we're done
                if not added_paper or not source_papers:
                    break
        
        logger.info(f"Balanced results: {len(balanced_results)} total papers from multiple sources")
        
        # Log the distribution by source
        source_counts = {}
        for paper in balanced_results:
            source = getattr(paper, "source", "unknown")
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
        
        logger.info(f"Results by source: {source_counts}")
        
        return balanced_results

    def _normalize_title(self, title):
        """Normalize title for duplicate detection"""
        import re
        # Remove punctuation, lowercase, and remove common words
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        words = normalized.split()
        # Remove very common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        filtered = [w for w in words if w not in stopwords]
        return ' '.join(filtered)

    def search(self, query):
        """
        Search for papers matching the query using both local index and external APIs.
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of relevant Paper objects
        """
        if query is None or not query.strip():
            logger.warning("Empty query received")
            return []
        
        logger.info(f"Searching for papers matching query: '{query}'")
        
        # Skip local search and rely on external sources
        external_results = self.search_external(query)
        
        # If no results, try fallback search
        if not external_results:
            logger.info(f"No results found for '{query}', trying fallback search")
            external_results = self.fallback_search(query)
        
        # Remove duplicates based on title
        unique_papers = []
        seen_titles = set()
        
        for paper in external_results:
            title_lower = paper.title.lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
                
        logger.info(f"Found {len(unique_papers)} total unique papers")
        return unique_papers
        
    def _search_local(self, query):
        """
        Search for papers matching the query using both BM25 and FAISS.
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of relevant Paper objects
        """
        # Your existing search code here
        # BM25 search - always available
        bm25_results = self._bm25_search(query, top_k=5)
        
        # FAISS search - only if available
        faiss_results = self._faiss_search(query, top_k=5) if FAISS_AVAILABLE and self.faiss_index is not None else []
        
        # Combine results (simple approach - union of both result sets)
        combined_indices = list(set(bm25_results + faiss_results))
        
        if not combined_indices:
            logger.info("No local results found for query")
            return []
        
        # Make sure we have valid indices
        valid_indices = [i for i in combined_indices if 0 <= i < len(self.papers)]
        
        results = [self.papers[i] for i in valid_indices]
        logger.info(f"Found {len(results)} papers matching query in local index")
        return results
    
    def _bm25_search(self, query, top_k=5):
        """Search using BM25 algorithm"""
        if self.bm25 is None or not self.papers:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Set minimum score threshold to consider a match
        min_score = 0.1
        
        # Get top results that meet minimum score
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [idx for idx in top_indices if scores[idx] > min_score]
    
    def _faiss_search(self, query, top_k=5):
        """Search using FAISS vector similarity"""
        if self.faiss_index is None or not self.papers:
            return []
        
        # Get query embedding
        query_embedding = self.text_embedder.generate_embeddings([query])
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Convert to Python list and return only valid indices
        return indices[0].tolist()
    
    def retrieve(self, paper_id):
        """
        Retrieve a specific paper by ID.
        
        Args:
            paper_id (int): The index of the paper to retrieve
            
        Returns:
            Paper or None: The requested Paper object or None if not found
        """
        try:
            return self.papers[paper_id]
        except IndexError:
            return None
    
    def execute(self, query):
        """
        Execute the search operation based on the query.
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of relevant Paper objects
        """
        return self.search(query)
    
    def respond(self):
        """Format and return the response"""
        # This would be used in a more complex agent interaction scenario
        pass

    def learn_from_feedback(self, query, relevant_papers, irrelevant_papers):
        """
        Update search model based on user feedback about results
        
        Args:
            query (str): The original query
            relevant_papers (list): Papers marked as relevant by the user
            irrelevant_papers (list): Papers marked as irrelevant by the user
        """
        if not relevant_papers:
            logger.warning("No relevant papers provided for learning")
            return
            
        logger.info(f"Learning from feedback for query: '{query}'")
        logger.info(f"Relevant papers: {len(relevant_papers)}, Irrelevant: {len(irrelevant_papers)}")
        
        try:
            # Extract text from papers for training
            relevant_texts = [f"{p.title} {p.abstract}" for p in relevant_papers]
            irrelevant_texts = [f"{p.title} {p.abstract}" for p in irrelevant_papers] if irrelevant_papers else []
            
            # Create or update a personalized BM25 model for this query pattern
            # In a real-world system, you might use more sophisticated ML here
            
            # For now, we'll just save these papers as preferred for this query type
            query_pattern = self._extract_query_pattern(query)
            
            # Persist this learning
            self._save_query_preferences(query_pattern, relevant_papers, irrelevant_papers)
            
            logger.info(f"Successfully learned from feedback for query pattern: {query_pattern}")
        except Exception as e:
            logger.error(f"Error learning from feedback: {str(e)}")
            
    def _extract_query_pattern(self, query):
        """Extract a generalizable pattern from the query"""
        # Simple implementation - lowercase and sort terms
        terms = query.lower().split()
        return " ".join(sorted(terms))
        
    def _save_query_preferences(self, query_pattern, relevant_papers, irrelevant_papers):
        """Save query preferences for future use"""
        preferences_path = os.path.join(os.path.dirname(self.index_path), "query_preferences.pkl")
        
        try:
            # Load existing preferences
            preferences = {}
            if os.path.exists(preferences_path):
                with open(preferences_path, 'rb') as f:
                    preferences = pickle.load(f)
            
            # Extract paper IDs
            relevant_ids = [p.title for p in relevant_papers]  # Using title as ID for simplicity
            irrelevant_ids = [p.title for p in irrelevant_papers]
            
            # Update preferences
            if query_pattern not in preferences:
                preferences[query_pattern] = {"relevant": [], "irrelevant": []}
                
            preferences[query_pattern]["relevant"].extend(relevant_ids)
            preferences[query_pattern]["irrelevant"].extend(irrelevant_ids)
            
            # Remove duplicates
            preferences[query_pattern]["relevant"] = list(set(preferences[query_pattern]["relevant"]))
            preferences[query_pattern]["irrelevant"] = list(set(preferences[query_pattern]["irrelevant"]))
            
            # Save updated preferences
            with open(preferences_path, 'wb') as f:
                pickle.dump(preferences, f)
                
            logger.info(f"Saved query preferences for pattern: {query_pattern}")
        except Exception as e:
            logger.error(f"Error saving query preferences: {str(e)}")

    def get_all_papers(self):
        """Return all papers in the index"""
        return self.papers

    def fallback_search(self, query):
        """
        Fallback search method when regular search returns no results
        
        Args:
            query (str): Search query
            
        Returns:
            list: Paper objects
        """
        logger.info(f"Using fallback search for query: {query}")
        
        # 1. Try with query expansion
        terms = query.lower().split()
        if len(terms) <= 3:  # Only for short queries
            related_terms = {
                "green": ["sustainable", "environmental", "ecology"],
                "finance": ["investment", "banking", "economic"],
                "climate": ["environment", "warming", "temperature"],
                "learning": ["education", "training", "knowledge"],
                "intelligence": ["cognitive", "thinking", "reasoning"],
                "quantum": ["particle", "physics", "mechanics"],
                # Add more related terms here
            }
            
            expanded_terms = []
            for term in terms:
                if term in related_terms:
                    expanded_terms.extend(related_terms[term][:2])  # Add up to 2 related terms
            
            if expanded_terms:
                expanded_query = query + " " + " ".join(expanded_terms)
                expanded_results = self.search_external(expanded_query)
                if expanded_results:
                    logger.info(f"Fallback search with expanded query returned {len(expanded_results)} results")
                    return expanded_results
        
        # 2. Try with broader categorization
        broader_query = " ".join([self._get_broader_category(term) for term in terms])
        if broader_query != query:
            broader_results = self.search_external(broader_query)
            if broader_results:
                logger.info(f"Fallback search with broader query returned {len(broader_results)} results")
                return broader_results
        
        # 3. Last resort: return sample data or empty list
        logger.warning("All fallback strategies failed, returning empty results")
        return []

    def _get_broader_category(self, term):
        """Convert specific term to broader category"""
        categories = {
            "green finance": "sustainable finance",
            "blockchain": "distributed technology",
            "bitcoin": "cryptocurrency",
            # Add more mappings
        }
        
        # Check if any key contains this term
        for specific, broader in categories.items():
            if term in specific:
                return broader
        
        return term