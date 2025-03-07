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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search/")
def search_papers(query: str, user_id: str = "anonymous"):
    """
    Search for papers matching the query
    
    Args:
        query (str): The search query
        user_id (str): User identifier for personalization
        
    Returns:
        dict: Search results
    """
    # Apply query improvements from learning
    improved_query = learning_agent.improve_query(query)
    
    # Get base results
    base_results = query_agent.process_query(improved_query)
    
    # Apply personalization if user_id is provided
    if user_id != "anonymous":
        personalized_results = learning_agent.get_personalized_results(user_id, query, base_results)
        return {"results": personalized_results, "improved_query": improved_query if improved_query != query else None}
    
    return {"results": base_results, "improved_query": improved_query if improved_query != query else None}

# Add feedback endpoint
@app.post("/feedback/")
def submit_feedback(user_id: str, query: str, paper_ids: list, clicked: list, time_spent: float, ratings: dict = None):
    """Record user feedback for learning"""
    # Convert paper IDs back to papers
    papers = [Paper(id=pid, title=pid) for pid in paper_ids]  # Simplified
    clicked_papers = [Paper(id=pid, title=pid) for pid in clicked]
    
    learning_agent.record_user_interaction(
        user_id=user_id,
        query=query,
        results=papers,
        clicked_papers=clicked_papers,
        time_spent=time_spent,
        explicit_feedback=ratings
    )
    
    return {"status": "success"}

# Add recommendations endpoint
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

# Run the server directly when this file is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
