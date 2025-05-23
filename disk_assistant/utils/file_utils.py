import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
import json

from config import CONFIG
from utils.logger import get_logger

logger = get_logger("FileUtils")


def get_file_hash(file_path: str, algorithm: str = "md5") -> Optional[str]:
    """
    Generate hash for a file to detect duplicates or changes.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

    Returns:
        Hash string or None if error
    """
    try:
        hash_func = getattr(hashlib, algorithm.lower())()

        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return None


def get_file_info(file_path: str) -> Optional[Dict]:
    """
    Get comprehensive file information.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information or None if error
    """
    try:
        path_obj = Path(file_path)

        if not path_obj.exists():
            return None

        stats = path_obj.stat()
        mime_type, encoding = mimetypes.guess_type(file_path)

        return {
            "path": str(path_obj.absolute()),
            "name": path_obj.name,
            "stem": path_obj.stem,
            "suffix": path_obj.suffix,
            "parent": str(path_obj.parent),
            "size": stats.st_size,
            "size_human": format_file_size(stats.st_size),
            "created": datetime.fromtimestamp(stats.st_ctime),
            "modified": datetime.fromtimestamp(stats.st_mtime),
            "accessed": datetime.fromtimestamp(stats.st_atime),
            "mime_type": mime_type,
            "encoding": encoding,
            "is_file": path_obj.is_file(),
            "is_dir": path_obj.is_dir(),
            "is_symlink": path_obj.is_symlink(),
            "permissions": oct(stats.st_mode)[-3:],
            "hash_md5": get_file_hash(file_path, "md5") if path_obj.is_file() else None,
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Convert bytes to human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def is_supported_file(file_path: str) -> bool:
    """
    Check if file type is supported based on configuration.

    Args:
        file_path: Path to the file

    Returns:
        True if file type is supported
    """
    path_obj = Path(file_path)
    return path_obj.suffix.lower() in CONFIG["file_extensions"]


def should_ignore_file(file_path: str) -> bool:
    """
    Check if file should be ignored based on ignore patterns.

    Args:
        file_path: Path to the file

    Returns:
        True if file should be ignored
    """
    file_str = str(file_path).lower()
    return any(pattern.lower() in file_str for pattern in CONFIG["ignore_patterns"])


def find_files(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
    include_hidden: bool = False,
) -> List[str]:
    """
    Find files in directory with optional filtering.

    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        recursive: Whether to search subdirectories
        include_hidden: Whether to include hidden files

    Returns:
        List of file paths
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory not found: {directory}")
            return []

        pattern = "**/*" if recursive else "*"
        all_files = dir_path.glob(pattern)

        results = []
        for file_path in all_files:
            if not file_path.is_file():
                continue

            # Skip hidden files if not requested
            if not include_hidden and file_path.name.startswith("."):
                continue

            # Check extensions if specified
            if extensions and file_path.suffix.lower() not in [
                ext.lower() for ext in extensions
            ]:
                continue

            # Check ignore patterns
            if should_ignore_file(str(file_path)):
                continue

            results.append(str(file_path))

        return sorted(results)
    except Exception as e:
        logger.error(f"Error finding files in {directory}: {e}")
        return []


def find_duplicates(directory: str, algorithm: str = "md5") -> Dict[str, List[str]]:
    """
    Find duplicate files based on hash comparison.

    Args:
        directory: Directory to search for duplicates
        algorithm: Hash algorithm to use

    Returns:
        Dictionary with hash as key and list of duplicate file paths as value
    """
    try:
        hash_map = {}
        duplicates = {}

        files = find_files(directory, recursive=True)

        for file_path in files:
            file_hash = get_file_hash(file_path, algorithm)
            if file_hash:
                if file_hash in hash_map:
                    if file_hash not in duplicates:
                        duplicates[file_hash] = [hash_map[file_hash]]
                    duplicates[file_hash].append(file_path)
                else:
                    hash_map[file_hash] = file_path

        return duplicates
    except Exception as e:
        logger.error(f"Error finding duplicates in {directory}: {e}")
        return {}


def safe_copy(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Safely copy a file with error handling.

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            logger.error(f"Source file not found: {src}")
            return False

        if dst_path.exists() and not overwrite:
            logger.warning(f"Destination exists and overwrite=False: {dst}")
            return False

        # Create destination directory if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)
        logger.info(f"File copied: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Error copying file {src} to {dst}: {e}")
        return False


def safe_move(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Safely move a file with error handling.

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            logger.error(f"Source file not found: {src}")
            return False

        if dst_path.exists() and not overwrite:
            logger.warning(f"Destination exists and overwrite=False: {dst}")
            return False

        # Create destination directory if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(src, dst)
        logger.info(f"File moved: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Error moving file {src} to {dst}: {e}")
        return False


def safe_delete(file_path: str, confirm: bool = True) -> bool:
    """
    Safely delete a file with optional confirmation.

    Args:
        file_path: Path to file to delete
        confirm: Whether to require confirmation (for interactive use)

    Returns:
        True if successful, False otherwise
    """
    try:
        path_obj = Path(file_path)

        if not path_obj.exists():
            logger.warning(f"File not found for deletion: {file_path}")
            return True  # Consider it successful if file doesn't exist

        if confirm:
            # In a real implementation, you might want to add user confirmation
            # For now, we'll log the action
            logger.info(f"Deleting file: {file_path}")

        path_obj.unlink()
        logger.info(f"File deleted: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def create_backup(file_path: str, backup_dir: Optional[str] = None) -> Optional[str]:
    """
    Create a backup of a file.

    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (defaults to same directory)

    Returns:
        Path to backup file or None if error
    """
    try:
        src_path = Path(file_path)

        if not src_path.exists():
            logger.error(f"Source file not found for backup: {file_path}")
            return None

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{src_path.stem}_backup_{timestamp}{src_path.suffix}"

        if backup_dir:
            backup_path = Path(backup_dir) / backup_name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            backup_path = src_path.parent / backup_name

        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        return str(backup_path)
    except Exception as e:
        logger.error(f"Error creating backup for {file_path}: {e}")
        return None


def get_directory_stats(directory: str) -> Optional[Dict]:
    """
    Get statistics about a directory.

    Args:
        directory: Directory path

    Returns:
        Dictionary with directory statistics
    """
    try:
        dir_path = Path(directory)

        if not dir_path.exists() or not dir_path.is_dir():
            return None

        total_files = 0
        total_dirs = 0
        total_size = 0
        file_types = {}

        for item in dir_path.rglob("*"):
            if item.is_file():
                total_files += 1
                try:
                    size = item.stat().st_size
                    total_size += size

                    ext = item.suffix.lower()
                    if ext not in file_types:
                        file_types[ext] = {"count": 0, "size": 0}
                    file_types[ext]["count"] += 1
                    file_types[ext]["size"] += size
                except (OSError, IOError):
                    # Skip files we can't access
                    continue
            elif item.is_dir():
                total_dirs += 1

        return {
            "path": str(dir_path.absolute()),
            "total_files": total_files,
            "total_directories": total_dirs,
            "total_size": total_size,
            "total_size_human": format_file_size(total_size),
            "file_types": file_types,
            "analyzed_at": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error getting directory stats for {directory}: {e}")
        return None


def cleanup_temp_files(temp_dir: str = "temp", max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.

    Args:
        temp_dir: Temporary directory to clean
        max_age_hours: Maximum age in hours before deletion

    Returns:
        Number of files deleted
    """
    try:
        temp_path = Path(temp_dir)

        if not temp_path.exists():
            return 0

        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        deleted_count = 0

        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old temp file: {file_path}")
                except (OSError, IOError) as e:
                    logger.debug(f"Could not delete temp file {file_path}: {e}")

        logger.info(f"Cleaned up {deleted_count} temporary files")
        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")
        return 0


def validate_file_path(file_path: str) -> Tuple[bool, str]:
    """
    Validate if a file path is safe and accessible.

    Args:
        file_path: Path to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path_obj = Path(file_path)

        # Check if path is absolute or can be resolved
        try:
            resolved_path = path_obj.resolve()
        except (OSError, IOError) as e:
            return False, f"Cannot resolve path: {e}"

        # Check if path exists
        if not resolved_path.exists():
            return False, "Path does not exist"

        # Check if we have read access
        if not os.access(resolved_path, os.R_OK):
            return False, "No read permission"

        # Check if it's in an ignored pattern
        if should_ignore_file(str(resolved_path)):
            return False, "File matches ignore pattern"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"
