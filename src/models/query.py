class Query:
    def __init__(self, query_text, user_id=None):
        self.query_text = query_text
        self.user_id = user_id
        self.timestamp = self.get_timestamp()

    def get_timestamp(self):
        from datetime import datetime
        return datetime.now()

    def __repr__(self):
        return f"Query(query_text='{self.query_text}', user_id='{self.user_id}', timestamp='{self.timestamp}')"