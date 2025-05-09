"""Initialization utilities for the application"""
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import setup_logger

logger = setup_logger("initialization")

def ensure_directories():
    """Create all required directories for the application"""
    project_root = Path(__file__).parent.parent.parent
    
    directories = [
        project_root / "data",
        project_root / "logs",
        project_root / "templates",
        project_root / "static" / "css",
        project_root / "static" / "js",
        project_root / "static" / "img"
    ]
    
    # Create all directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    
    return {
        "data_dir": project_root / "data",
        "templates_dir": project_root / "templates",
        "static_dir": project_root / "static"
    }

def initialize_environment():
    """Load environment variables and return configurations"""
    # Load environment variables
    load_dotenv()
    
    # Create paths dictionary
    paths = ensure_directories()
    
    # Build config dictionary
    config = {
        "data_path": str(paths["data_dir"]),
        "index_path": str(paths["data_dir"] / "paper_index"),
        "templates_dir": str(paths["templates_dir"]),
        "static_dir": str(paths["static_dir"]),
        "max_results": int(os.getenv("MAX_RESULTS_PER_SOURCE", "10"))
    }
    
    logger.info("Environment initialized successfully")
    return config