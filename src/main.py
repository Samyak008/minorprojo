from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from agents.retrieval_agent import RetrievalAgent
from agents.query_agent import QueryAgent
import os
from pathlib import Path
import uvicorn

app = FastAPI()

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search/")
def search_papers(query: str):
    results = query_agent.process_query(query)
    return {"results": results}

@app.get("/health/")
def health_check():
    return {"status": "healthy"}

# Run the server directly when this file is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)