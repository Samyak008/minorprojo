# Research Paper Retrieval System

A multi-agent AI platform for searching, personalizing, and recommending research papers from multiple sources (e.g., arXiv, Semantic Scholar, CORE, Crossref, OpenAlex, etc.). This system uses FastAPI for the backend, Sentence-Transformers for embedding and similarity, and BM25 or FAISS for indexing locally stored paper metadata.

## Table of Contents
1. [Features](#1-features)
2. [Directory Structure](#2-directory-structure)
3. [System Overview](#3-system-overview)
4. [Setup and Installation](#4-setup-and-installation)
5. [Environment Variables](#5-environment-variables)
6. [Running the Application](#6-running-the-application)
7. [Usage and Endpoints](#7-usage-and-endpoints)
8. [Testing](#8-testing)
9. [Extending the System](#9-extending-the-system)
10. [License](#10-license)


## 1. Features
- **Multi-agent architecture:**
  - **Retrieval Agent:** Performs local text indexing with BM25 or FAISS and fetches from external APIs.
  - **Query Agent:** Enhances user queries, removes duplicates, and orchestrates the final result.
  - **Learning Agent:** Analyzes user feedback and query success to perform continuous updates and personalization.
- **Integration with multiple external APIs** (arXiv, Semantic Scholar, CORE, Crossref, OpenAlex) for up-to-date research.
- **Personalization logic** based on user interactions (e.g., clicks, time spent), stored in the Learning Agent.
- **Web front-end** served by FastAPI (using a "templates" folder) with a simple search interface.
- **Example tests** for retrieval and query functionality in the "tests" folder.

## 2. Directory Structure
```
D:\minorprojo
│
├── .env
├── requirements.txt
│
├── src
│   ├── __init__.py
│   ├── main.py               # FastAPI entry point
│   │
│   ├── agents                # Intelligent agents
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── query_agent.py
│   │   ├── retrieval_agent.py
│   │   └── learning_agent.py
│   │
│   ├── models
│   │   ├── __init__.py
│   │   └── paper.py
│   │
│   ├── utils
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py
│   │   ├── semantic_scholar_fetcher.py
│   │   ├── core_fetcher.py
│   │   ├── crossref_fetcher.py
│   │   ├── openalex_fetcher.py
│   │   ├── ieee_fetcher.py
│   │   ├── springer_fetcher.py
│   │   └── text_embeddings.py
│   │
│   └── templates             # HTML templates for FastAPI
│       └── index.html
│
└── tests
    ├── __init__.py
    ├── test_query_agent.py
    └── test_retrieval_agent.py
```

## 3. System Overview
1. User visits the main page (`GET /`), served from the `templates` folder.
2. User enters a search query, and the frontend calls `POST /search/?query=...`.
3. The request is routed to:
   - **Learning Agent** for query improvements.
   - **Query Agent** to process the query.
   - **Retrieval Agent** to fetch relevant papers from local indexes or external APIs.
4. The Learning Agent applies personalization if a `user_id` is provided.
5. The results are displayed, and user feedback can be used for personalization.

## 4. Setup and Installation
### Clone Repository
```sh
git clone <repository_url>
cd minorprojo
```
### Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
```
### Install Dependencies
```sh
pip install -r requirements.txt
```

## 5. Environment Variables
Create a `.env` file in the root directory with:
```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
IEEE_API_KEY=your_key_here
CORE_API_KEY=your_key_here
SPRINGER_API_KEY=your_key_here
MAX_RESULTS_PER_SOURCE=10
```

## 6. Running the Application
Activate the virtual environment (if used) and start the FastAPI server:
```sh
uvicorn src.main:app --reload
```
Access at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 7. Usage and Endpoints
### `GET /`
Renders the main landing page.

### `POST /search/?query=<query>&user_id=<optional_user_id>`
Main search endpoint. Returns:
```json
{
  "improved_query": "<updated_query>",
  "results": [<list_of_papers>]
}
```

### `POST /feedback/`
Accepts JSON feedback from users:
```json
{
  "user_id": "user123",
  "query": "machine learning",
  "clicked_papers": ["paper1", "paper2"]
}
```
Returns:
```json
{"message": "Feedback received"}
```

### `GET /recommendations/{paper_id}`
Returns recommended papers similar to `paper_id`.

### `GET /health/`
Health check endpoint returning JSON:
```json
{"status": "OK"}
```

## 8. Testing
Run tests using:
```sh
pytest tests/
```

## 9. Extending the System
### Adding a New Fetcher
1. Create `src/utils/some_fetcher.py`.
2. Implement a `search_papers(self, query)` method.
3. Register the fetcher in `retrieval_agent.py`.

### Improving Learning Logic
Modify `LearningAgent` for advanced ML-based personalization.

## 10. License
(Include license details, e.g., MIT, Apache 2.0, etc.)

