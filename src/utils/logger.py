"""Centralized logging configuration"""
import logging
import os
import sys
from pathlib import Path

def setup_logger(name=None, level=logging.INFO):
    """Set up and return a logger with consistent configuration"""
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)
    
    # Only configure logger if it hasn't been configured before
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create logs directory if it doesn't exist
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Setup file handler
        file_handler = logging.FileHandler(logs_dir / "app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default application logger
app_logger = setup_logger("research_paper_retrieval")