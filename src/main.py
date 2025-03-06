from fastapi import FastAPI
from agents.retrieval_agent import RetrievalAgent
from agents.query_agent import QueryAgent

app = FastAPI()

retrieval_agent = RetrievalAgent()
query_agent = QueryAgent(retrieval_agent)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Research Paper Retrieval System"}

@app.post("/search/")
def search_papers(query: str):
    results = query_agent.process_query(query)
    return {"results": results}

@app.get("/health/")
def health_check():
    return {"status": "healthy"}