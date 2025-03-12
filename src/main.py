from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from agents.retrieval_agent import RetrievalAgent
from agents.query_agent import QueryAgent
from agents.learning_agent import LearningAgent
from models.paper import Paper
import os
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
from rag_gym import RAGGym

app = FastAPI()

# Load environment variables
load_dotenv()

# Define index path relative to project root
data_dir = Path(__file__).parent.parent / "data"
index_path = str(data_dir / "paper_index")

# Create templates directory if it doesn't exist
templates_dir = Path(__file__).parent.parent / "templates"
os.makedirs(templates_dir, exist_ok=True)

# Create a simple HTML template if it doesn't exist
index_html_path = templates_dir / "index.html"
if not index_html_path.exists():
    with open(index_html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Paper Retrieval System</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                h1 { color: #333; }
                .container { max-width: 800px; margin: 0 auto; }
                .search-form { margin: 20px 0; }
                input[type="text"] { padding: 8px; width: 70%; font-size: 16px; }
                button { padding: 8px 16px; background: #4CAF50; color: white; border: none; cursor: pointer; font-size: 16px; }
                .results { margin-top: 20px; }
                .paper { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
                .paper h3 { margin-top: 0; color: #2962FF; }
                .authors { color: #555; font-style: italic; }
                .abstract { margin-top: 10px; }
                .year { color: #757575; font-weight: bold; }
                .loading { display: none; margin-top: 20px; color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Research Paper Retrieval System</h1>
                <div class="search-form">
                    <input type="text" id="search-input" placeholder="Enter your search query...">
                    <button onclick="searchPapers()">Search</button>
                </div>
                <div id="loading" class="loading">Searching papers...</div>
                <div id="results" class="results"></div>
            </div>

            <script>
                async function searchPapers() {
                    const query = document.getElementById('search-input').value.trim();
                    if (!query) return;

                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('results').innerHTML = '';

                    try {
                        const response = await fetch(`/search/?query=${encodeURIComponent(query)}`, {
                            method: 'POST'
                        });
                        const data = await response.json();

                        document.getElementById('loading').style.display = 'none';

                        if (data.results && data.results.length > 0) {
                            const resultsContainer = document.getElementById('results');
                            resultsContainer.innerHTML = `<h2>Found ${data.results.length} papers:</h2>`;

                            data.results.forEach(paper => {
                                const paperElement = document.createElement('div');
                                paperElement.className = 'paper';
                                paperElement.innerHTML = `
                                    <h3>${paper.title}</h3>
                                    <div class="authors">By ${paper.authors.join(', ')}</div>
                                    <div class="abstract">${paper.abstract}</div>
                                    <div class="year">Published in ${paper.publication_year}</div>
                                `;
                                resultsContainer.appendChild(paperElement);
                            });
                        } else {
                            document.getElementById('results').innerHTML = '<p>No papers found matching your query.</p>';
                        }
                    } catch (error) {
                        console.error('Error searching papers:', error);
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('results').innerHTML = '<p>Error searching papers. Please try again.</p>';
                    }
                }
            </script>
        </body>
        </html>
        """)

templates = Jinja2Templates(directory=str(templates_dir))

# Ensure directory exists
os.makedirs(data_dir, exist_ok=True)

retrieval_agent = RetrievalAgent(index_path=index_path)
query_agent = QueryAgent(retrieval_agent)
learning_agent = LearningAgent(retrieval_agent, query_agent)

# Add RAG-Gym initialization after the other agents
rag_gym = RAGGym(
    retrieval_agent=retrieval_agent,
    query_agent=query_agent,
    learning_agent=learning_agent
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Modify the search endpoint to use RAG-Gym's step-wise search
@app.post("/search/")
def search_papers(query: str, user_id: str = "anonymous"):
    """
    Search for papers using RAG-Gym's step-wise retrieval logic
    
    Args:
        query (str): The search query
        user_id (str): User identifier for personalization
        
    Returns:
        dict: Search results
    """
    # Start RAG-Gym session
    session_id = rag_gym.start_session(user_id, query)
    
    try:
        # Perform all steps in one go instead of separate calls
        final_result = rag_gym.complete_search(session_id)
        return final_result
    except Exception as e:
        logger.error(f"RAG-Gym search error: {str(e)}")
        # Fall back to standard search
        improved_query = learning_agent.improve_query(query)
        base_results = query_agent.process_query(improved_query)
        return {
            "results": base_results,
            "improved_query": improved_query if improved_query != query else None
        }

# Update feedback endpoint to utilize RAG-Gym's reward modeling
@app.post("/feedback/")
def submit_feedback(user_id: str, query: str, paper_ids: list, clicked: list, time_spent: float, ratings: dict = None):
    """Record user feedback for Process Reward Modeling training"""
    # Convert paper IDs to papers once
    papers = {pid: Paper(id=pid, title=pid) for pid in set(paper_ids + clicked)}
    
    # Create paper lists
    result_papers = [papers[pid] for pid in paper_ids]
    clicked_papers = [papers[pid] for pid in clicked]
    
    # Record feedback in both systems in parallel
    learning_agent.record_user_interaction(
        user_id=user_id, query=query, results=result_papers,
        clicked_papers=clicked_papers, time_spent=time_spent, 
        explicit_feedback=ratings
    )
    
    rag_gym.record_feedback(
        user_id=user_id, query=query, results=result_papers,
        clicked_papers=clicked_papers, feedback=ratings
    )
    
    return {"status": "success"}

# Add RAG-Gym training endpoints
@app.post("/train/prm/")
def train_prm_model(epochs: int = 10):
    """Train Process Reward Model"""
    success = rag_gym.train_prm_reward_model(epochs)
    return {"status": "success" if success else "failed"}

@app.post("/train/sft/")
def train_sft_model(epochs: int = 10):
    """Train Supervised Fine-Tuning model"""
    success = rag_gym.train_sft_model(epochs)
    return {"status": "success" if success else "failed"}

# Add explanation endpoint that leverages RAG capabilities
@app.get("/explain/{paper_id}")
def explain_paper_relevance(paper_id: str, query: str):
    """Explain why a paper is relevant to the query"""
    paper = Paper(id=paper_id, title=paper_id)
    paper_dict = query_agent._paper_to_dict(paper)
    
    if not hasattr(rag_gym, "_calculate_query_relevance"):
        return {"error": "Explanation functionality not available"}
        
    # Calculate relevance directly
    relevance_score = rag_gym._calculate_query_relevance(paper_dict, query)
    
    # Simplify term matching
    query_terms = set(query.lower().split())
    title_terms = set(paper_dict.get("title", "").lower().split())
    abstract_terms = set(paper_dict.get("abstract", "").lower().split())
    
    matches = {
        "title": list(query_terms.intersection(title_terms)),
        "abstract": list(query_terms.intersection(abstract_terms))
    }
    
    return {
        "relevance_score": relevance_score,
        "matches": matches,
        "explanation": f"Relevance score: {relevance_score:.2f}. Found {len(matches['title'])} title matches and {len(matches['abstract'])} abstract matches."
    }

@app.get("/recommendations/{paper_id}")
def get_recommendations(paper_id: str):
    """Get recommendations for similar papers"""
    # In a real implementation, you'd retrieve the full paper
    paper = Paper(id=paper_id, title=paper_id)
    
    recommendations = learning_agent.recommend_related_papers(paper)
    return {"recommendations": [query_agent._paper_to_dict(p) for p in recommendations]}

@app.get("/health/")
def health_check():
    return {"status": "healthy"}

# Add session-based RAG-Gym API endpoints
@app.post("/rag/sessions/")
def create_session(query: str, user_id: str = "anonymous"):
    """Create a new RAG-Gym search session"""
    session_id = rag_gym.start_session(user_id, query)
    return {"session_id": session_id}

@app.post("/rag/sessions/{session_id}/step")
def process_session_step(session_id: str):
    """Process the next step in the RAG-Gym search session"""
    result = rag_gym.step_wise_search(session_id)
    return result

@app.get("/rag/sessions/{session_id}")
def get_session_status(session_id: str):
    """Get the current status of a RAG-Gym search session"""
    if session_id not in rag_gym.active_sessions:
        return {"error": "Session not found"}
        
    session = rag_gym.active_sessions[session_id]
    return {
        "user_id": session["user_id"],
        "query": session["original_query"],
        "step": session["step"],
        "results_count": len(session.get("results", []))
    }

# Run the server directly when this file is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
