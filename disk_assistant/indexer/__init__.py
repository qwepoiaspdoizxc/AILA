"""
Indexer module for the Disk Assistant.
Handles file indexing, monitoring, and search functionality.
"""

from .file_index import FileIndex
from .monitor import FileSystemMonitor

__all__ = ["FileIndex", "FileSystemMonitor"]
