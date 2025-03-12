import requests
import time
import logging
import schedule
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_training")

# Server URL
BASE_URL = "http://localhost:8000"

def train_prm_model():
    """Train Process Reward Model"""
    try:
        logger.info("Starting PRM model training")
        response = requests.post(f"{BASE_URL}/train/prm/", params={"epochs": 20})
        
        if response.status_code == 200 and response.json().get("status") == "success":
            logger.info("PRM model training completed successfully")
        else:
            logger.error(f"PRM model training failed: {response.text}")
    except Exception as e:
        logger.error(f"Error training PRM model: {str(e)}")

def train_sft_model():
    """Train Supervised Fine-Tuning model"""
    try:
        logger.info("Starting SFT model training")
        response = requests.post(f"{BASE_URL}/train/sft/", params={"epochs": 15})
        
        if response.status_code == 200 and response.json().get("status") == "success":
            logger.info("SFT model training completed successfully")
        else:
            logger.error(f"SFT model training failed: {response.text}")
    except Exception as e:
        logger.error(f"Error training SFT model: {str(e)}")

def run_all_training():
    """Run all training jobs"""
    logger.info("Running scheduled model training")
    train_prm_model()
    train_sft_model()
    
    # Write training timestamp
    try:
        data_dir = Path(__file__).parent.parent / "data" / "rag_gym"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_file = data_dir / "last_training.json"
        with open(timestamp_file, 'w') as f:
            json.dump({
                "last_training": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f)
    except Exception as e:
        logger.error(f"Error saving training timestamp: {str(e)}")

if __name__ == "__main__":
    # Schedule training to run daily at 3 AM
    schedule.every().day.at("03:00").do(run_all_training)
    
    logger.info("Training scheduler started. Will run daily at 3 AM.")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute