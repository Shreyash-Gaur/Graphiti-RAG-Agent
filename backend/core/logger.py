import logging
import sys
from pathlib import Path

# Define log paths
LOG_DIR = Path("backend/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

def setup_logging():
    """
    Configures the ROOT logger. 
    This ensures that all modules (backend.main, backend.agents, etc.) 
    automatically log to the console and the file.
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Prevent adding duplicate handlers if setup_logging is called twice
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (Print to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Save to backend/logs/app.log)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    return logger

def get_logger(name: str):
    """
    Helper to get a named logger. 
    Since setup_logging configures the root, these will inherit the handlers.
    """
    return logging.getLogger(name)