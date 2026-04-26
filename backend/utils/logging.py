import logging
import sys
from pathlib import Path

def setup_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Ensure log directory exists
    log_dir = Path("backend/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "bullrun_api.log")
        ]
    )
    
    # Silence third-party loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger("bullrun_api")
