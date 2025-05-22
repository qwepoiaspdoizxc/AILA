import os
import json
from pathlib import Path
from utils.logger import logger

# Enhanced configuration dictionary for AI-powered assistant
CONFIG = {
    # Existing file indexing configuration
    "index_dir": "file_index",
    "db_path": "file_metadata.db",
    "directories_to_watch": ["C:\\Users\\ccagubcub\\Documents\\Training"],
    "file_extensions": [
        ".txt", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".odt", ".rtf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif",
        ".tiff", ".ico", ".svg", ".webp", ".mp3", ".wav", ".wma", ".aac",
        ".flac", ".ogg", ".m4a", ".mp4", ".avi", ".mkv", ".mov", ".wmv",
        ".flv", ".webm", ".mpeg", ".3gp", ".zip", ".rar", ".7z", ".tar",
        ".gz", ".bz2", ".xz", ".iso", ".cab", ".exe", ".msi", ".bat",
        ".cmd", ".com", ".ps1", ".vbs", ".dll", ".scr", ".py", ".java",
        ".c", ".cpp", ".cs", ".js", ".ts", ".html", ".css", ".json",
        ".xml", ".php", ".rb", ".go", ".rs", ".sql", ".sh", ".yml",
        ".yaml", ".ini", ".cfg", ".asm", ".sys", ".drv", ".inf", ".reg",
        ".bak", ".log", ".tmp", ".ttf", ".otf", ".fon", ".fnt", ".db",
        ".dbf", ".mdb", ".accdb", ".sav", ".map", ".swf", ".torrent",
        ".lnk", ".crt", ".pem", ".csv"
    ],
    
    # Voice recognition configuration
    "vosk_model_path": "C:/Users/ccagubcub/Documents/AudioRecognition/vosk-model-en-us-0.22",
    "wake_words": ["hey assistant", "disk assistant", "computer", "ai assistant"],
    "activation_threshold": 0.7,
    "voice_timeout": 5.0,  # seconds to wait for command after wake word
    "continuous_listening": True,
    
    # AI Model configuration
    "ai_model_path": "models/local_llm",  # Path to local LLM (e.g., GGUF format)
    "ai_model_type": "llama",  # Support for different model types
    "max_context_length": 4096,
    "temperature": 0.7,
    "max_tokens": 512,
    
    # Audio recording configuration
    "recordings_dir": "recordings",
    "audio_format": "wav",
    "sample_rate": 44100,
    "channels": 2,
    "recording_quality": "high",
    "auto_transcription": True,
    
    # Data analysis configuration
    "analysis_cache_dir": "analysis_cache",
    "chart_output_dir": "charts",
    "supported_data_formats": [".csv", ".xlsx", ".json", ".parquet"],
    "max_analysis_file_size": 100 * 1024 * 1024,  # 100MB
    
    # Enhanced scanning configuration
    "scan_interval": 3600,
    "ignore_patterns": [
        "__pycache__", ".git", ".vscode", "node_modules", ".DS_Store",
        "Thumbs.db", "*.tmp", "*.temp", ".env", "venv", ".venv"
    ],
    "max_file_size": 50 * 1024 * 1024,  # Increased to 50MB
    
    # AI Response configuration
    "response_modes": ["voice", "text", "both"],
    "default_response_mode": "both",
    "voice_speed": 150,  # words per minute
    "voice_engine": "system",  # system, espeak, or festival
    
    # Privacy and security
    "local_processing_only": True,
    "encrypt_recordings": False,
    "auto_delete_recordings": False,
    "retention_days": 30,
    
    # Performance settings
    "max_concurrent_operations": 3,
    "index_batch_size": 100,
    "ai_processing_threads": 2
}


def load_config():
    """Load configuration from file or create default"""
    config_path = "ai_disk_assistant_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                # Update global CONFIG
                for key, value in loaded_config.items():
                    CONFIG[key] = value
            logger.info(f"Loaded AI configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    else:
        # Save default config
        try:
            with open(config_path, "w") as f:
                json.dump(CONFIG, f, indent=2)
            logger.info(f"Created default AI configuration at {config_path}")
        except Exception as e:
            logger.error(f"Error creating default config: {e}")


def save_config():
    """Save current configuration to file"""
    config_path = "ai_disk_assistant_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=2)
        logger.info(f"AI Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def ensure_directories():
    """Ensure all required directories exist"""
    dirs_to_create = [
        CONFIG["index_dir"],
        CONFIG["recordings_dir"],
        CONFIG["analysis_cache_dir"],
        CONFIG["chart_output_dir"],
        "logs",
        "temp"
    ]
    
    for directory in dirs_to_create:
        Path(directory).mkdir(exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check critical paths
    if not os.path.exists(CONFIG["vosk_model_path"]):
        issues.append(f"Vosk model not found at: {CONFIG['vosk_model_path']}")
    
    # Check directories to watch
    for directory in CONFIG["directories_to_watch"]:
        if not os.path.exists(directory):
            issues.append(f"Watch directory not found: {directory}")
    
    # Check AI model path if specified
    if CONFIG.get("ai_model_path") and not os.path.exists(CONFIG["ai_model_path"]):
        issues.append(f"AI model not found at: {CONFIG['ai_model_path']}")
    
    if issues:
        logger.warning("Configuration validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return len(issues) == 0