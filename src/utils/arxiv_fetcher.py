import arxiv
import logging
from models.paper import Paper

logger = logging.getLogger(__name__)

class ArxivFetcher:
    def __init__(self, max_results=20):
        self.max_results = max_results
        
    def search_papers(self, query):
        """
        Search for papers on ArXiv matching the query
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of Paper objects
        """
        try:
            logger.info(f"Searching ArXiv for: {query}")
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in search.results():
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    publication_year=result.published.strftime("%Y"),
                    url=result.entry_id,
                    source="arxiv"
                )
                results.append(paper)
            
            logger.info(f"Found {len(results)} papers on ArXiv")
            return results
        except Exception as e:
            logger.error(f"Error searching ArXiv: {str(e)}")
            return []