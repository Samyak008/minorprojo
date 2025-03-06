class Paper:
    def __init__(self, title, authors, abstract, publication_year):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.publication_year = publication_year

    def __repr__(self):
        return f"Paper(title={self.title}, authors={self.authors}, abstract={self.abstract}, publication_year={self.publication_year})"