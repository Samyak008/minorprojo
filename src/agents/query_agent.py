import logging
import os
import json

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
        
        # Check if query needs domain-specific enhancement
        enhanced_query = self._enhance_domain_query(query)
        if enhanced_query != query:
            logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
        
        # Use the retrieval agent to search for papers
        papers = self.retrieval_agent.search(enhanced_query)
        
        # Convert paper objects to dictionaries
        paper_dicts = [self._paper_to_dict(paper) for paper in papers]
        
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
        """
        Apply learned transformations to improve the query
        
        Args:
            query (str): Original query
            
        Returns:
            str: Improved query
        """
        # Check if we have a direct transformation
        if query in self._query_transformations:
            improved = self._query_transformations[query]
            logger.info(f"Applied direct transformation: '{query}' -> '{improved}'")
            return improved
        
        # Look for similar queries
        for known_query, transformation in self._query_transformations.items():
            if self._queries_are_similar(query, known_query):
                # Apply similar transformation
                improved = self._apply_similar_transformation(query, known_query, transformation)
                logger.info(f"Applied similar transformation: '{query}' -> '{improved}'")
                return improved
        
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