import os
import json
from pathlib import Path
from utils.logger import logger

# Configuration dictionary
CONFIG = {
    "index_dir": "file_index",
    "db_path": "file_metadata.db",
    "directories_to_watch": ["C:\\Users\\ccagubcub\\Documents\\Training"],
    "file_extensions": [
        ".txt",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".odt",
        ".rtf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".wav",
        ".wma",
        ".aac",
        ".flac",
        ".ogg",
        ".m4a",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".mpeg",
        ".3gp",
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".iso",
        ".cab",
        ".exe",
        ".msi",
        ".bat",
        ".cmd",
        ".com",
        ".ps1",
        ".vbs",
        ".dll",
        ".scr",
        ".py",
        ".java",
        ".c",
        ".cpp",
        ".cs",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".php",
        ".rb",
        ".go",
        ".rs",
        ".sql",
        ".sh",
        ".yml",
        ".yaml",
        ".ini",
        ".cfg",
        ".asm",
        ".sys",
        ".drv",
        ".inf",
        ".reg",
        ".bak",
        ".log",
        ".tmp",
        ".ttf",
        ".otf",
        ".fon",
        ".fnt",
        ".db",
        ".dbf",
        ".mdb",
        ".accdb",
        ".sav",
        ".map",
        ".swf",
        ".torrent",
        ".lnk",
        ".crt",
        ".pem",
    ],
    "vosk_model_path": "C:/Users/ccagubcub/Documents/AudioRecognition/vosk-model-en-us-0.22",
    "scan_interval": 3600,
    "ignore_patterns": [
        "__pycache__",
        ".git",
        ".vscode",
        "node_modules",
        ".DS_Store",
        "Thumbs.db",
    ],
    "max_file_size": 10 * 1024 * 1024 * 1024,  # Maximum file size to index (10MB)
}


def load_config():
    """Load configuration from file or create default"""
    config_path = "disk_assistant_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                # Update global CONFIG
                for key, value in loaded_config.items():
                    CONFIG[key] = value
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    else:
        # Save default config
        try:
            with open(config_path, "w") as f:
                json.dump(CONFIG, f, indent=2)
            logger.info(f"Created default configuration at {config_path}")
        except Exception as e:
            logger.error(f"Error creating default config: {e}")


def save_config():
    """Save current configuration to file"""
    config_path = "disk_assistant_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False
