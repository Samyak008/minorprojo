import requests
import logging
from models.paper import Paper

logger = logging.getLogger(__name__)

class SemanticScholarFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": api_key} if api_key else {}
        
    def search_papers(self, query, limit=20):
        """
        Search for papers on Semantic Scholar matching the query
        
        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
            
        Returns:
            list: A list of Paper objects
        """
        try:
            logger.info(f"Searching Semantic Scholar for: {query}")
            endpoint = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,year,url"
            }
            
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("data", []):
                if not item.get("title") or not item.get("abstract"):
                    continue
                    
                paper = Paper(
                    title=item.get("title", ""),
                    authors=[author.get("name", "") for author in item.get("authors", [])],
                    abstract=item.get("abstract", ""),
                    publication_year=str(item.get("year", "")),
                    url=item.get("url", ""),
                    source="semantic_scholar"
                )
                results.append(paper)
            
            logger.info(f"Found {len(results)} papers on Semantic Scholar")
            return results
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return []