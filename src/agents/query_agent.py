import logging
import os
import json
import re
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

class QueryAgent:
    def __init__(self, retrieval_agent):
        """
        Initialize the QueryAgent with a reference to the RetrievalAgent.
        
        Args:
            retrieval_agent: The RetrievalAgent instance to use for retrieving papers
        """
        self.retrieval_agent = retrieval_agent
        self._query_transformations = {}
        self._load_query_patterns()
    
    def process_query(self, query):
        """
        Process a user query and return relevant paper results.
        
        Args:
            query (str): The user's search query
        
        Returns:
            list: A list of relevant Paper objects converted to dictionaries
        """
        logger.info(f"Processing query: '{query}'")
        
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []
        
        # Use ReSearch approach for dynamic query refinement
        paper_dicts, refined_query = self.process_query_with_research(query)
        
        # Remove duplicates based on title
        unique_papers = []
        seen_titles = set()
        for paper in paper_dicts:
            if paper['title'].lower() not in seen_titles:
                seen_titles.add(paper['title'].lower())
                unique_papers.append(paper)
        
        logger.info(f"Found {len(unique_papers)} unique papers for query '{query}'")
        return unique_papers
    
    def _paper_to_dict(self, paper):
        """
        Convert a Paper object to a dictionary for JSON serialization.
        
        Args:
            paper: A Paper object
        
        Returns:
            dict: Dictionary representation of the paper
        """
        return {
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "publication_year": paper.publication_year,
            "url": getattr(paper, "url", None),
            "source": getattr(paper, "source", None),
            "doi": getattr(paper, "doi", None),
            "citations": getattr(paper, "citations", None)
        }

    def learn_query_patterns(self, original_query, successful_query, results_quality_score):
        """
        Learn to improve queries based on successful searches
        
        Args:
            original_query (str): User's original query
            successful_query (str): Query that yielded good results
            results_quality_score (float): Measure of result quality (0-1)
        """
        if results_quality_score > 0.7 and original_query != successful_query:
            # Store this query transformation for future use
            self._query_transformations[original_query] = successful_query
            self._save_query_patterns()

    def _load_query_patterns(self):
        """Load previously learned query patterns"""
        patterns_path = "data/query_patterns.json"
        try:
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r') as f:
                    self._query_transformations = json.load(f)
                logger.info(f"Loaded {len(self._query_transformations)} query patterns")
        except Exception as e:
            logger.error(f"Error loading query patterns: {str(e)}")

    def _save_query_patterns(self):
        """Save learned query patterns"""
        patterns_path = "data/query_patterns.json"
        try:
            os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
            with open(patterns_path, 'w') as f:
                json.dump(self._query_transformations, f)
            logger.info(f"Saved {len(self._query_transformations)} query patterns")
        except Exception as e:
            logger.error(f"Error saving query patterns: {str(e)}")

    def improve_query(self, query):
        """Apply learned transformations to improve the query"""
        # Check if we have a direct transformation
        if query in self._query_transformations:
            return self._query_transformations[query]
        
        # Process all known queries at once to find similar ones
        similar_queries = [(known_query, transformation) 
                           for known_query, transformation in self._query_transformations.items()
                           if self._queries_are_similar(query, known_query)]
        
        # Apply transformation from the most similar query
        if similar_queries:
            known_query, transformation = similar_queries[0]
            return self._apply_similar_transformation(query, known_query, transformation)
        
        # No similar query found, return original
        return query

    def _queries_are_similar(self, query1, query2):
        """Check if two queries are semantically similar"""
        # Simple implementation - check for common terms
        terms1 = set(query1.lower().split())
        terms2 = set(query2.lower().split())
        
        if not terms1 or not terms2:
            return False
        
        common = terms1.intersection(terms2)
        similarity = len(common) / max(len(terms1), len(terms2))
        
        return similarity > 0.5

    def _apply_similar_transformation(self, query, known_query, known_transformation):
        """Apply a transformation pattern from a similar query"""
        # Simple implementation - add terms from the known transformation
        query_terms = set(query.lower().split())
        known_terms = set(known_query.lower().split())
        transformation_terms = set(known_transformation.lower().split())
        
        # Add terms that were added in the known transformation
        added_terms = transformation_terms - known_terms
        
        if added_terms:
            return f"{query} {' '.join(added_terms)}"
        
        return query

    def _enhance_domain_query(self, query):
        """Add domain-specific terms to improve search relevance"""
        query_lower = query.lower()
        
        # Domain-specific enhancements
        enhancements = {
            "green finance": "green finance sustainable investment ESG climate",
            "climate": "climate change global warming environmental",
            "ai": "artificial intelligence machine learning neural networks",
            "ml": "machine learning algorithms data science",
            "nlp": "natural language processing computational linguistics",
            "quantum": "quantum computing quantum mechanics qubits",
            "blockchain": "blockchain distributed ledger cryptocurrency",
            "cybersecurity": "cybersecurity information security privacy",
            "iot": "internet of things IoT smart devices connected",
            "robotics": "robotics automation autonomous systems",
            "renewable": "renewable energy sustainable solar wind",
        }
        
        # Check if the query contains any of our domain keywords
        for key, enhancement in enhancements.items():
            if key in query_lower:
                # Only add terms that aren't already in the query
                extra_terms = []
                for term in enhancement.split():
                    if term.lower() not in query_lower:
                        extra_terms.append(term)
                
                if extra_terms:
                    return f"{query} {' '.join(extra_terms)}"
        
        return query

    def dynamic_query_refinement(self, original_query, max_iterations=2):
        """Implement ReSearch approach for dynamic query refinement"""
        logger.info(f"Starting dynamic query refinement for: '{original_query}'")
        
        current_query = original_query
        best_query = original_query
        best_results = []
        best_score = 0
        
        for iteration in range(max_iterations):
            # Get search results for current query
            results = self.retrieval_agent.search(current_query)
            
            if not results:
                break
                
            # Score the results quality
            score = self._evaluate_results_quality(results, original_query)
            
            # Update best results if better
            if score > best_score:
                best_score = score
                best_query = current_query
                best_results = results
                
            # Stop if we have good results or query converged
            if score > 0.8:
                break
                
            # Generate a refined query based on results
            refined_query = self._refine_query_from_results(current_query, results, original_query)
            if refined_query == current_query:
                break
                
            current_query = refined_query
        
        # Record successful transformation for learning
        if best_query != original_query and best_score > 0.5:
            self.learn_query_patterns(original_query, best_query, best_score)
            
        return best_query, best_results

    def _evaluate_results_quality(self, results, original_query):
        """
        Evaluate the quality of search results for a given query
        
        Args:
            results (list): List of paper results
            original_query (str): The original query
            
        Returns:
            float: Quality score between 0 and 1
        """
        if not results:
            return 0.0
            
        # Extract query terms
        query_terms = set(self._tokenize_and_normalize(original_query))
        if not query_terms:
            return 0.5  # Neutral score for empty query
        
        # Calculate relevance metrics
        title_matches = []
        abstract_matches = []
        recency_scores = []
        
        current_year = datetime.now().year
        
        for paper in results:
            # Title relevance - what percentage of query terms appear in the title?
            title_terms = set(self._tokenize_and_normalize(paper.title))
            title_match = len(query_terms.intersection(title_terms)) / len(query_terms)
            title_matches.append(title_match)
            
            # Abstract relevance
            abstract_terms = set(self._tokenize_and_normalize(paper.abstract))
            abstract_match = len(query_terms.intersection(abstract_terms)) / len(query_terms)
            abstract_matches.append(abstract_match)
            
            # Recency - more recent papers score higher
            try:
                year = int(paper.publication_year) if paper.publication_year else 0
                recency = 1.0 if year >= current_year - 3 else 0.7 if year >= current_year - 10 else 0.4
            except (ValueError, TypeError):
                recency = 0.5  # Default if year parsing fails
            
            recency_scores.append(recency)
        
        # Calculate overall score with different weights
        avg_title_match = sum(title_matches) / len(title_matches)
        avg_abstract_match = sum(abstract_matches) / len(abstract_matches)
        avg_recency = sum(recency_scores) / len(recency_scores)
        
        # Title matches are most important, followed by abstract matches and recency
        overall_score = (0.5 * avg_title_match) + (0.3 * avg_abstract_match) + (0.2 * avg_recency)
        
        return overall_score

    def _refine_query_from_results(self, current_query, results, original_query):
        """
        Generate a refined query based on search results
        
        Args:
            current_query (str): Current query
            results (list): Current search results
            original_query (str): Original user query
            
        Returns:
            str: Refined query
        """
        # Always keep original query terms
        original_terms = set(self._tokenize_and_normalize(original_query))
        
        # Extract key terms from top results
        top_results = results[:5]  # Use top 5 results for query refinement
        
        # Get frequent terms from titles
        title_terms = []
        for paper in top_results:
            title_terms.extend(self._tokenize_and_normalize(paper.title))
        
        # Count term frequencies
        term_counts = Counter(title_terms)
        
        # Remove terms already in query
        for term in self._tokenize_and_normalize(current_query):
            if term in term_counts:
                del term_counts[term]
        
        # Get top new terms (most frequent)
        new_terms = [term for term, count in term_counts.most_common(2) if count > 1]
        
        # Build refined query: original terms + new informative terms
        if new_terms:
            all_terms = set(self._tokenize_and_normalize(current_query))
            all_terms.update(new_terms)
            # Always ensure original query terms are included
            all_terms.update(original_terms)
            return " ".join(all_terms)
        
        return current_query

    def _tokenize_and_normalize(self, text):
        """
        Tokenize and normalize text for comparing terms
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: Normalized tokens
        """
        if not text:
            return []
            
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and short terms
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}
        return [token for token in tokens if token not in stopwords and len(token) > 2]

    def process_query_with_research(self, query):
        """
        Process query using ReSearch approach for dynamic query refinement
        
        Args:
            query (str): Original user query
            
        Returns:
            list: Paper dictionaries for results
        """
        # Apply domain-specific enhancement first
        enhanced_query = self._enhance_domain_query(query)
        
        # Apply Direct Preference Optimization if learned patterns exist
        improved_query = self.improve_query(enhanced_query)
        
        # Apply ReSearch dynamic query refinement
        refined_query, results = self.dynamic_query_refinement(improved_query)
        
        # Convert paper objects to dictionaries
        paper_dicts = [self._paper_to_dict(paper) for paper in results]
        
        # Log the query transformation
        if refined_query != query:
            logger.info(f"Query transformation: '{query}' -> '{refined_query}'")
        
        return paper_dicts, refined_query