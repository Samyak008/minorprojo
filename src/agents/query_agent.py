import logging

logger = logging.getLogger(__name__)

class QueryAgent:
    def __init__(self, retrieval_agent):
        """
        Initialize the QueryAgent with a reference to the RetrievalAgent.
        
        Args:
            retrieval_agent: The RetrievalAgent instance to use for retrieving papers
        """
        self.retrieval_agent = retrieval_agent
    
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
        
        # Use the retrieval agent to search for papers
        papers = self.retrieval_agent.search(query)
        
        # Convert paper objects to dictionaries
        paper_dicts = [self._paper_to_dict(paper) for paper in papers]
        
        # Remove duplicates based on title
        unique_papers = []
        seen_titles = set()
        for paper in paper_dicts:
            if paper['title'] not in seen_titles:
                seen_titles.add(paper['title'])
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