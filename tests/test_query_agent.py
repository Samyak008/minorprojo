from src.agents.query_agent import QueryAgent
from src.agents.retrieval_agent import RetrievalAgent

def test_query_agent_initialization():
    retrieval_agent = RetrievalAgent()
    query_agent = QueryAgent(retrieval_agent)
    assert query_agent.retrieval_agent == retrieval_agent

def test_process_query():
    retrieval_agent = RetrievalAgent()
    query_agent = QueryAgent(retrieval_agent)
    
    query = "machine learning in healthcare"
    expected_results = retrieval_agent.search(query)
    
    results = query_agent.process_query(query)
    assert results == expected_results

def test_empty_query():
    retrieval_agent = RetrievalAgent()
    query_agent = QueryAgent(retrieval_agent)
    
    query = ""
    results = query_agent.process_query(query)
    assert results == []  # Expecting no results for an empty query

def test_invalid_query_format():
    retrieval_agent = RetrievalAgent()
    query_agent = QueryAgent(retrieval_agent)
    
    query = "12345"  # Assuming this is an invalid query format
    results = query_agent.process_query(query)
    assert results == []  # Expecting no results for an invalid query format