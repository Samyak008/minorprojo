class Paper:
    def __init__(self, title, authors, abstract, publication_year, url=None, source=None, doi=None, citations=None):
        """
        Initialize a Paper object.
        
        Args:
            title (str): The paper title
            authors (list): List of author names
            abstract (str): The paper abstract
            publication_year (str): The publication year
            url (str, optional): URL to the paper
            source (str, optional): Source of the paper (arxiv, semantic_scholar, etc.)
            doi (str, optional): Digital Object Identifier
            citations (int, optional): Number of citations
        """
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.publication_year = publication_year
        self.url = url
        self.source = source
        self.doi = doi
        self.citations = citations

    def __repr__(self):
        return f"Paper(title={self.title}, authors={self.authors}, abstract={self.abstract}, publication_year={self.publication_year})"