import requests
import logging
from models.paper import Paper

logger = logging.getLogger(__name__)

class OpenAlexFetcher:
    def __init__(self, max_results=20):
        self.max_results = max_results
        self.base_url = "https://api.openalex.org/works"
        # Use your email for polite API use
        self.headers = {
            "User-Agent": "ResearchPaperRetrievalSystem/1.0 (your.email@example.com)"
        }
        
    def search_papers(self, query):
        """
        Search for papers on OpenAlex matching the query
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of Paper objects
        """
        if not query:
            return []
            
        try:
            logger.info(f"Searching OpenAlex for: {query}")
            
            params = {
                "search": query,
                "per_page": self.max_results,
                "filter": "has_abstract:true"
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.warning(f"OpenAlex API returned status code {response.status_code}")
                return []
                
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                # Extract authors
                authors = []
                for author in item.get("authorships", []):
                    if "author" in author and "display_name" in author["author"]:
                        authors.append(author["author"]["display_name"])
                
                # Extract year
                year = ""
                publication_date = item.get("publication_date")
                if publication_date:
                    year = publication_date.split("-")[0]
                
                # Extract DOI
                doi = ""
                if "doi" in item and item["doi"]:
                    doi = item["doi"].replace("https://doi.org/", "")
                
                # Create Paper object
                paper = Paper(
                    title=item.get("title", ""),
                    authors=authors,
                    abstract=item.get("abstract", ""),
                    publication_year=year,
                    url=item.get("doi") or item.get("id", ""),
                    source="openalex",
                    doi=doi,
                    citations=item.get("cited_by_count", 0)
                )
                
                if paper.title and paper.abstract:  # Only include if it has both title and abstract
                    results.append(paper)
            
            logger.info(f"Found {len(results)} papers on OpenAlex")
            return results
            
        except Exception as e:
            logger.error(f"Error searching OpenAlex: {str(e)}")
            return []