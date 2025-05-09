from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from agents.retrieval_agent import RetrievalAgent
from agents.query_agent import QueryAgent
from agents.learning_agent import LearningAgent
from models.paper import Paper
import os
import logging
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
from rag_gym import RAGGym  # Import the RAG-Gym class  
from typing import Dict, List, Optional
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create FastAPI instance
app = FastAPI(
    title="Research Paper Retrieval System", 
    description="An AI-powered academic paper search engine with RAG capabilities"
)

# Set up configuration
load_dotenv()

# Directory structure setup
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
static_dir = project_root / "static"
templates_dir = project_root / "templates"

# Create directories if they don't exist
directories = [
    data_dir,
    static_dir,
    static_dir / "css",
    static_dir / "js",
    static_dir / "img",
    templates_dir
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Initialize file paths
index_path = str(data_dir / "paper_index")

# Create CSS file if it doesn't exist
css_path = static_dir / "css" / "style.css"

# Create JS file if it doesn't exist
js_path = static_dir / "js" / "app.js"

# Create HTML template if it doesn't exist
index_html_path = templates_dir / "index.html"

# Initialize agents
try:
    # Initialize retrieval agent
    retrieval_agent = RetrievalAgent(index_path=index_path)
    
    # Initialize query agent with retrieval agent
    query_agent = QueryAgent(retrieval_agent=retrieval_agent)
    
    # Initialize learning agent with both agents
    learning_agent = LearningAgent(
        retrieval_agent=retrieval_agent,
        query_agent=query_agent,
        data_dir=data_dir / "learning"
    )
    
    # Initialize RAG-Gym with all agents
    rag_gym = RAGGym(
        retrieval_agent=retrieval_agent,
        query_agent=query_agent,
        learning_agent=learning_agent,
        data_dir=data_dir / "rag_gym"
    )
    logger.info("Agents initialized successfully")
except Exception as e:
    logger.error(f"Error initializing agents: {str(e)}")
    raise

# Utility functions
def paper_to_dict(paper):
    """Convert Paper object to dictionary with safe handling of missing attributes"""
    return {
        "id": getattr(paper, "id", ""),
        "title": getattr(paper, "title", "Untitled"),
        "authors": getattr(paper, "authors", []),
        "abstract": getattr(paper, "abstract", ""),
        "publication_year": getattr(paper, "publication_year", None),
        "source": getattr(paper, "source", None),
        "citations": getattr(paper, "citations", None),
        "doi": getattr(paper, "doi", None),
        "url": getattr(paper, "url", None)
    }

def create_papers_from_ids(paper_ids):
    """Create paper objects from IDs efficiently"""
    return {pid: Paper(id=pid, title=pid) for pid in set(paper_ids)}

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Configure templates
templates = Jinja2Templates(directory=str(templates_dir))

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search/")
def search_papers(query: str, user_id: str = "anonymous"):
    """Search for papers using RAG-Gym's retrieval logic"""
    logger.info(f"Search request from user {user_id}: '{query}'")
    
    try:
        # Start RAG-Gym session and perform complete search
        session_id = rag_gym.start_session(user_id, query)
        result = rag_gym.complete_search(session_id)
        
        # Add session ID to result for tracking
        result["session_id"] = session_id
        return result
    except Exception as e:
        logger.error(f"Error in RAG-Gym search: {str(e)}")
        
        # Fall back to standard search
        try:
            # Use learning agent to improve query
            improved_query = learning_agent.improve_query(query)
            
            # Use query agent to process the improved query
            results = query_agent.process_query(improved_query)
            
            return {
                "results": results,
                "improved_query": improved_query if improved_query != query else None
            }            # Update the search_papers function to include metrics
            
            @app.post("/search/")
            def search_papers(query: str, user_id: str = "anonymous"):
                """Search for papers using RAG workflow"""
                logger.info(f"Search request from user {user_id}: '{query}'")
                
                try:
                    # Start timing the request
                    start_time = time.time()
                    
                    # Start and execute RAG workflow
                    workflow_id = rag_workflow.start_workflow(user_id, query)
                    result = rag_workflow.execute_workflow(workflow_id)
                    
                    # Ensure metrics are included in the response
                    if "metrics" not in result:
                        elapsed = time.time() - start_time
                        result["metrics"] = {
                            "total_time_seconds": round(elapsed, 2),
                            "raw_results_count": len(result.get("results", [])),
                            "final_results_count": len(result.get("results", []))
                        }
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in RAG workflow search: {str(e)}")
                    
                    # Fall back to standard search
                    try:
                        # Use learning agent to improve query
                        improved_query = learning_agent.improve_query(query)
                        
                        # Use query agent to process the improved query
                        results = query_agent.process_query(improved_query)
                        
                        # Calculate elapsed time
                        elapsed = time.time() - start_time
                        
                        return {
                            "results": results,
                            "improved_query": improved_query if improved_query != query else None,
                            "metrics": {
                                "total_time_seconds": round(elapsed, 2),
                                "raw_results_count": len(results),
                                "final_results_count": len(results)
                            }
                        }
                    except Exception as e2:
                        logger.error(f"Error in fallback search: {str(e2)}")
                        raise HTTPException(status_code=500, detail=f"Search error: {str(e2)}")
        except Exception as e2:
            logger.error(f"Error in fallback search: {str(e2)}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e2)}")

@app.post("/feedback/")
async def submit_feedback(user_id: str, query: str, paper_ids: List[str], 
                          clicked: List[str], time_spent: float, ratings: Optional[Dict] = None):
    """Record user feedback for training"""
    try:
        # Efficiently create paper objects using utility function
        papers = create_papers_from_ids(paper_ids + clicked)
        
        # Create paper lists from the dictionary
        result_papers = [papers[pid] for pid in paper_ids]
        clicked_papers = [papers[pid] for pid in clicked]
        
        # Process feedback in background to improve response time
        def process_feedback():
            try:
                # Record in both systems
                learning_agent.record_user_interaction(
                    user_id=user_id, 
                    query=query, 
                    results=result_papers,
                    clicked_papers=clicked_papers, 
                    time_spent=time_spent, 
                    explicit_feedback=ratings
                )
                
                rag_gym.record_feedback(
                    user_id=user_id,
                    query=query,
                    results=result_papers,
                    clicked_papers=clicked_papers,
                    feedback=ratings
                )
            except Exception as e:
                logger.error(f"Error processing feedback: {str(e)}")
                
        # Run feedback processing in background
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(process_feedback)
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/explain/{paper_id}")
def explain_paper_relevance(paper_id: str, query: str):
    """Explain why a paper is relevant to the query"""
    try:
        # Create a Paper object from the ID
        paper = Paper(id=paper_id, title=paper_id)
        
        # Convert to dictionary for processing
        paper_dict = paper_to_dict(paper)
        
        # Calculate relevance using RAG-Gym's utility function
        if not hasattr(rag_gym, "_calculate_query_relevance"):
            return {"error": "Explanation functionality not available"}
            
        relevance_score = rag_gym._calculate_query_relevance(paper_dict, query)
        
        # Extract matched terms between query and paper
        query_terms = set(query.lower().split())
        title_terms = set(paper_dict.get("title", "").lower().split())
        abstract_terms = set(paper_dict.get("abstract", "").lower().split())
        
        # Calculate term matches
        matches = {
            "title": list(query_terms.intersection(title_terms)),
            "abstract": list(query_terms.intersection(abstract_terms))
        }
        
        # Generate explanation
        explanation = (
            f"This paper has a relevance score of {relevance_score:.2f} for your query. "
            f"It contains {len(matches['title'])} matching terms in the title and "
            f"{len(matches['abstract'])} matching terms in the abstract."
        )
        
        return {
            "relevance_score": relevance_score,
            "matches": matches,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error explaining relevance: {str(e)}")
        return {"error": str(e)}

@app.get("/recommendations/{paper_id}")
async def get_recommendations(paper_id: str, limit: int = 5):
    """Get papers similar to the specified paper"""
    try:
        paper = Paper(id=paper_id, title=paper_id)
        similar_papers = learning_agent.recommend_related_papers(paper, limit=limit)
        return {"recommendations": [paper_to_dict(p) for p in similar_papers]}
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return {"error": str(e)}

@app.get("/rag/sessions/{session_id}")
def get_session_status(session_id: str):
    """Get the current status of a RAG-Gym search session"""
    try:
        if session_id not in rag_gym.active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session = rag_gym.active_sessions[session_id]
        return {
            "user_id": session["user_id"],
            "query": session["original_query"],
            "step": session["step"],
            "results_count": len(session.get("results", [])),
            "elapsed_time": (datetime.now() - session.get("start_time", datetime.now())).total_seconds()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics/search")
async def record_search_metrics(metrics: Dict):
    """Record anonymous search metrics for system improvement"""
    try:
        # Log metrics for analysis
        logger.info(f"Search metrics: {json.dumps(metrics)}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error recording metrics: {str(e)}")
        return {"status": "error"}

@app.post("/rag/train")
async def train_rag_models():
    """Train RAG-Gym models using collected data"""
    try:
        prm_success = rag_gym.train_prm_reward_model()
        sft_success = rag_gym.train_sft_model()
        
        return {
            "status": "success",
            "prm_trained": prm_success,
            "sft_trained": sft_success
        }
    except Exception as e:
        logger.error(f"Error training RAG models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
