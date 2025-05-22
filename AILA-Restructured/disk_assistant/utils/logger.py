import logging
import os
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "disk_assistant.log"),
        logging.StreamHandler()
    ],
)

# Create the main logger
logger = logging.getLogger("DiskAssistant")

# Create specialized loggers for different modules
indexer_logger = logging.getLogger("DiskAssistant.Indexer")
monitor_logger = logging.getLogger("DiskAssistant.Monitor")
speech_logger = logging.getLogger("DiskAssistant.Speech")
nlp_logger = logging.getLogger("DiskAssistant.NLP")
ui_logger = logging.getLogger("DiskAssistant.UI")


def get_logger(name):
    """Get a logger for a specific module"""
    return logging.getLogger(f"DiskAssistant.{name}")


def set_log_level(level):
    """Set the logging level for all loggers"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.getLogger("DiskAssistant").setLevel(numeric_level)
    logger.info(f"Log level set to {level.upper()}")