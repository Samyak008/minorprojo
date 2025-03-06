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
        Search for papers using external APIs
        
        Args:
            query (str): The search query
            
        Returns:
            list: List of Paper objects from external sources
        """
        if query is None or not query.strip():
            return []
            
        logger.info(f"Searching external sources for: {query}")
        
        # Search ArXiv
        arxiv_results = self.arxiv_fetcher.search_papers(query)
        
        # Search Semantic Scholar
        semantic_scholar_results = self.semantic_scholar_fetcher.search_papers(query)
        
        # Combine results
        combined_results = arxiv_results + semantic_scholar_results
        
        # Remove duplicates based on title (simple approach)
        unique_papers = []
        seen_titles = set()
        
        for paper in combined_results:
            if paper.title not in seen_titles:
                seen_titles.add(paper.title)
                unique_papers.append(paper)
                
        logger.info(f"Found {len(unique_papers)} unique papers from external sources")
        return unique_papers

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
        
        # Local search
        local_results = self._search_local(query)
        
        # External search if local search returns few results
        external_results = []
        if len(local_results) < 5:  # Arbitrary threshold
            external_results = self.search_external(query)
        
        # Combine results
        all_results = local_results + external_results
        
        # Remove duplicates based on title
        unique_papers = []
        seen_titles = set()
        
        for paper in all_results:
            if paper.title not in seen_titles:
                seen_titles.add(paper.title)
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