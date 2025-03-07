import requests
import logging
import os
from models.paper import Paper
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class SpringerFetcher:
    def __init__(self, max_results=10):
        self.api_key = os.getenv("SPRINGER_API_KEY")
        if not self.api_key:
            logger.warning("Springer API key not found in environment variables")
        self.max_results = max_results
        self.base_url = "http://api.springernature.com/metadata/json"
        
    def search_papers(self, query):
        """
        Search for papers on Springer matching the query
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of Paper objects
        """
        if not self.api_key:
            logger.warning("Skipping Springer search due to missing API key")
            return []
            
        try:
            logger.info(f"Searching Springer for: {query}")
            
            params = {
                "api_key": self.api_key,
                "q": query,
                "p": self.max_results
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for record in data.get("records", []):
                # Create Paper object
                paper = Paper(
                    title=record.get("title", ""),
                    authors=[creator.get("creator", "") for creator in record.get("creators", [])],
                    abstract=record.get("abstract", ""),
                    publication_year=record.get("publicationDate", "")[:4] if record.get("publicationDate") else "",
                    url=record.get("url", {}).get("value", ""),
                    source="springer",
                    doi=record.get("doi", "")
                )
                results.append(paper)
            
            logger.info(f"Found {len(results)} papers on Springer")
            return results
        except Exception as e:
            logger.error(f"Error searching Springer: {str(e)}")
            return []