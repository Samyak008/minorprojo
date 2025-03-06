from fastapi import FastAPI
from agents.retrieval_agent import RetrievalAgent
from agents.query_agent import QueryAgent
import os
from pathlib import Path

app = FastAPI()

# Define index path relative to project root
data_dir = Path(__file__).parent.parent / "data"
index_path = str(data_dir / "paper_index")

# Ensure directory exists
os.makedirs(data_dir, exist_ok=True)

retrieval_agent = RetrievalAgent(index_path=index_path)
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