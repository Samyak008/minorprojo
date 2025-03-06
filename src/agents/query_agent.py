class QueryAgent:
    def __init__(self, retrieval_agent):
        self.retrieval_agent = retrieval_agent

    def process_query(self, user_query):
        parsed_query = self.parse_query(user_query)
        results = self.retrieval_agent.search(parsed_query)
        return results

    def parse_query(self, user_query):
        # Implement query parsing logic here
        return user_query.strip()  # Simple example of normalization