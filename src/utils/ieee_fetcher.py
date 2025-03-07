import requests
import logging
import os
from models.paper import Paper
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)

class IEEEFetcher:
    def __init__(self, max_results=10):
        self.api_key = os.getenv("IEEE_API_KEY")
        if not self.api_key:
            logger.warning("IEEE API key not found in environment variables")
        self.max_results = max_results
        self.base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"
        
    def search_papers(self, query):
        """
        Search for papers on IEEE Xplore matching the query
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of Paper objects
        """
        if not self.api_key:
            logger.warning("Skipping IEEE search due to missing API key")
            return []
            
        try:
            logger.info(f"Searching IEEE for: {query}")
            
            params = {
                "apikey": self.api_key,
                "format": "json",
                "max_records": self.max_results,
                "start_record": 1,
                "querytext": query,
                "sort_order": "relevant"
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for article in data.get("articles", []):
                # Extract authors
                authors = []
                for author in article.get("authors", {}).get("authors", []):
                    authors.append(f"{author.get('full_name', '')}")
                
                # Create Paper object
                paper = Paper(
                    title=article.get("title", ""),
                    authors=authors,
                    abstract=article.get("abstract", ""),
                    publication_year=article.get("publication_year", ""),
                    url=article.get("html_url", ""),
                    source="ieee",
                    doi=article.get("doi", ""),
                    citations=article.get("citing_paper_count", 0)
                )
                results.append(paper)
            
            logger.info(f"Found {len(results)} papers on IEEE")
            return results
        except Exception as e:
            logger.error(f"Error searching IEEE: {str(e)}")
            return []