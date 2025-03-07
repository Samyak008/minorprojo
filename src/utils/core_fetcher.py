import requests
import logging
import os
from models.paper import Paper
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class CoreFetcher:
    def __init__(self, max_results=20):
        self.api_key = os.getenv("CORE_API_KEY", "")  # CORE offers free API keys with registration
        self.max_results = max_results
        self.base_url = "https://api.core.ac.uk/v3"
        
    def search_papers(self, query):
        """
        Search for papers on CORE matching the query
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of Paper objects
        """
        if not query:
            return []
            
        try:
            logger.info(f"Searching CORE for: {query}")
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Even without API key, CORE allows limited access
            payload = {
                "q": query,
                "limit": self.max_results,
                "offset": 0
            }
            
            response = requests.post(
                f"{self.base_url}/search/works", 
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning(f"CORE API returned status code {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                # Extract authors
                authors = []
                for author in item.get("authors", []):
                    if isinstance(author, dict) and "name" in author:
                        authors.append(author["name"])
                    elif isinstance(author, str):
                        authors.append(author)
                
                # Extract year
                year = ""
                if "yearPublished" in item and item["yearPublished"]:
                    year = str(item["yearPublished"])
                
                # Create Paper object
                paper = Paper(
                    title=item.get("title", ""),
                    authors=authors,
                    abstract=item.get("abstract", "") or item.get("description", ""),
                    publication_year=year,
                    url=item.get("downloadUrl") or item.get("identifiers", {}).get("doi") or "",
                    source="core",
                    doi=item.get("doi", "")
                )
                
                if paper.title and paper.abstract:  # Only include if it has both title and abstract
                    results.append(paper)
            
            logger.info(f"Found {len(results)} papers on CORE")
            return results
            
        except Exception as e:
            logger.error(f"Error searching CORE: {str(e)}")
            return []