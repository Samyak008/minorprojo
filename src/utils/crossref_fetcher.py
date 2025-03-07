import requests
import logging
from models.paper import Paper
import time

logger = logging.getLogger(__name__)

class CrossrefFetcher:
    def __init__(self, max_results=20):
        self.max_results = max_results
        self.base_url = "https://api.crossref.org/works"
        # Use your email for polite API use
        self.headers = {
            "User-Agent": "ResearchPaperRetrievalSystem/1.0 (your.email@example.com)"
        }
        
    def search_papers(self, query):
        """
        Search for papers on Crossref matching the query
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of Paper objects
        """
        if not query:
            return []
            
        try:
            logger.info(f"Searching Crossref for: {query}")
            
            params = {
                "query": query,
                "rows": self.max_results,
                "sort": "relevance",
                "select": "DOI,title,abstract,author,published"
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.warning(f"Crossref API returned status code {response.status_code}")
                return []
                
            data = response.json()
            results = []
            
            for item in data.get("message", {}).get("items", []):
                # Extract authors
                authors = []
                for author in item.get("author", []):
                    name_parts = []
                    if "given" in author:
                        name_parts.append(author["given"])
                    if "family" in author:
                        name_parts.append(author["family"])
                    if name_parts:
                        authors.append(" ".join(name_parts))
                
                # Extract year
                year = ""
                if "published" in item and "date-parts" in item["published"]:
                    date_parts = item["published"]["date-parts"]
                    if date_parts and date_parts[0]:
                        year = str(date_parts[0][0])
                
                # Extract abstract
                abstract = item.get("abstract", "")
                
                # Sometimes the abstract is HTML-encoded
                abstract = abstract.replace("&lt;", "<").replace("&gt;", ">")
                # Remove HTML tags (simplified approach)
                abstract = abstract.replace("<p>", "").replace("</p>", " ").replace("<jats:p>", "").replace("</jats:p>", " ")
                
                # Create Paper object
                paper = Paper(
                    title=item.get("title", [""])[0] if item.get("title") else "",
                    authors=authors,
                    abstract=abstract,
                    publication_year=year,
                    url=f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else "",
                    source="crossref",
                    doi=item.get("DOI", "")
                )
                
                if paper.title:  # Only include if it has a title
                    results.append(paper)
            
            logger.info(f"Found {len(results)} papers on Crossref")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Crossref: {str(e)}")
            return []