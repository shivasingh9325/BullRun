import logging
import os
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    
    # File Handler
    # Use relative path or env variable
    logs_dir = os.getenv("LOGS_DIR", "logs")
    if not os.path.isabs(logs_dir):
        # Resolve relative to backend/ (assumes script is inside /bullrun/utils/)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir = os.path.join(base_dir, logs_dir)
        
    os.makedirs(logs_dir, exist_ok=True)
    f_handler = logging.FileHandler(os.path.join(logs_dir, "bullrun.log"))
    f_handler.setLevel(logging.INFO)
    
    # Formatter
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(c_format)
    
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger
