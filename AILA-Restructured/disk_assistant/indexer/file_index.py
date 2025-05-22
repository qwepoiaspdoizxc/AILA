import os
import sqlite3
from pathlib import Path
from datetime import datetime
import logging

from config import CONFIG

# Third-party imports
try:
    from whoosh.index import create_in, open_dir
    from whoosh.fields import Schema, TEXT, ID, DATETIME, STORED
    from whoosh.qparser import QueryParser, MultifieldParser
except ImportError:
    raise ImportError("whoosh module not found. Please install it with: pip install whoosh")

logger = logging.getLogger("DiskAssistant")


class FileIndex:
    """Manages file indexing using Whoosh and SQLite."""

    def __init__(self, index_dir=None, db_path=None):
        self.index_dir = index_dir or CONFIG["index_dir"]
        self.db_path = db_path or CONFIG["db_path"]
        self._setup_whoosh()
        self._setup_sqlite()

    def _setup_whoosh(self):
        """Initialize the Whoosh index."""
        schema = Schema(
            path=ID(stored=True, unique=True),
            filename=TEXT(stored=True),
            content=TEXT,
            file_type=TEXT(stored=True),
            last_modified=DATETIME(stored=True),
        )

        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            self.index = create_in(self.index_dir, schema)
            logger.info(f"Created new index at {self.index_dir}")
        else:
            try:
                self.index = open_dir(self.index_dir)
                logger.info(f"Opened existing index at {self.index_dir}")
            except Exception as e:
                # If index is corrupted, recreate it
                logger.warning(f"Index appears corrupted, recreating: {e}")
                import shutil
                shutil.rmtree(self.index_dir)
                os.makedirs(self.index_dir)
                self.index = create_in(self.index_dir, schema)

    def _setup_sqlite(self):
        """Initialize the SQLite database for file metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables if they don't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                filename TEXT,
                file_type TEXT,
                last_modified TIMESTAMP,
                file_size INTEGER,
                indexed_at TIMESTAMP
            )
            """)

            # Create indices for faster searching
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON files (filename)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON files (file_type)")

            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")

    def add_file(self, file_path):
        """Add or update a file in both Whoosh index and SQLite DB."""
        try:
            path_obj = Path(file_path)

            # Skip if file doesn't exist or isn't a supported type
            if not path_obj.exists() or path_obj.is_dir():
                return False

            # Skip if file extension is not in supported list
            if path_obj.suffix.lower() not in CONFIG["file_extensions"]:
                return False

            # Skip files in ignore patterns
            if any(ignore in str(path_obj) for ignore in CONFIG["ignore_patterns"]):
                return False

            # Get file metadata
            stats = path_obj.stat()

            # Skip if file is too large
            if stats.st_size > CONFIG["max_file_size"]:
                logger.debug(f"Skipping large file: {file_path} ({stats.st_size} bytes)")
                return False

            last_modified = datetime.fromtimestamp(stats.st_mtime)
            file_size = stats.st_size

            # Extract basic text content based on file type
            try:
                content = self._extract_text_content(file_path)
            except Exception as e:
                logger.debug(f"Content extraction error for {file_path}: {e}")
                content = ""  # Use empty content if extraction fails

            # Update Whoosh index
            try:
                writer = self.index.writer()
                writer.update_document(
                    path=str(file_path),
                    filename=path_obj.name,
                    content=content,
                    file_type=path_obj.suffix.lower(),
                    last_modified=last_modified,
                )
                writer.commit()
            except Exception as e:
                logger.debug(f"Whoosh indexing error for {file_path}: {e}")
                return False

            # Update SQLite DB
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                INSERT OR REPLACE INTO files
                (path, filename, file_type, last_modified, file_size, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(file_path),
                        path_obj.name,
                        path_obj.suffix.lower(),
                        last_modified,
                        file_size,
                        datetime.now(),
                    ),
                )
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                logger.debug(f"SQLite error for {file_path}: {e}")
                return False

            return True
        except (PermissionError, OSError, IOError) as e:
            logger.debug(f"File access error for {file_path}: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error indexing {file_path}: {e}")
            return False

    def _extract_text_content(self, file_path):
        """Extract text content from file based on type."""
        path_obj = Path(file_path)
        file_type = path_obj.suffix.lower()

        # For simplicity, just read text files directly
        if file_type == ".txt" or file_type == ".py":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if len(content) > 100000:  # Limit content size to 100KB
                    content = content[:100000]
                return content
            except Exception as e:
                logger.debug(f"Error reading {file_path}: {e}")
                return ""

        # For other file types, just use the filename as content for now
        return path_obj.name

    def remove_file(self, file_path):
        """Remove a file from index and database."""
        if not file_path or not os.path.exists(file_path):
            # Still try to remove from index and DB
            try:
                writer = self.index.writer()
                writer.delete_by_term("path", str(file_path))
                writer.commit()

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM files WHERE path = ?", (str(file_path),))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug(f"Error removing non-existent file {file_path}: {e}")
            return True

        try:
            # Remove from Whoosh
            writer = self.index.writer()
            writer.delete_by_term("path", str(file_path))
            writer.commit()

            # Remove from SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM files WHERE path = ?", (str(file_path),))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.debug(f"Error removing {file_path} from index: {e}")
            return False

    def search(self, query_text, limit=20):
        """Search for files matching the query."""
        results = []

        try:
            # Search in Whoosh index
            with self.index.searcher() as searcher:
                query_parser = MultifieldParser(["filename", "content"], self.index.schema)
                query = query_parser.parse(query_text)
                whoosh_results = searcher.search(query, limit=limit)

                for hit in whoosh_results:
                    results.append({
                        "path": hit["path"],
                        "filename": hit["filename"],
                        "file_type": hit["file_type"],
                        "last_modified": hit["last_modified"],
                        "score": hit.score,
                    })

            # If we have fewer than limit results, also search SQLite
            if len(results) < limit:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Simple filename search
                cursor.execute(
                    """
                SELECT * FROM files 
                WHERE filename LIKE ? 
                LIMIT ?
                """,
                    (f"%{query_text}%", limit - len(results)),
                )

                for row in cursor.fetchall():
                    # Skip if already in results
                    if any(r["path"] == row["path"] for r in results):
                        continue

                    results.append({
                        "path": row["path"],
                        "filename": row["filename"],
                        "file_type": row["file_type"],
                        "last_modified": datetime.fromtimestamp(row["last_modified"]),
                        "score": 0.5,  # Default score for DB matches
                    })

                conn.close()

            # Sort by score (highest first)
            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def scan_directory(self, directory):
        """Scan a directory and index all files."""
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory not found: {directory}")
            return 0

        total_files = 0
        indexed_files = 0

        for file_path in dir_path.glob("**/*"):
            # Skip directories and files in ignore patterns
            if file_path.is_dir() or any(
                ignore in str(file_path) for ignore in CONFIG["ignore_patterns"]
            ):
                continue

            total_files += 1
            if self.add_file(str(file_path)):
                indexed_files += 1

            # Periodically log progress
            if indexed_files % 100 == 0 and indexed_files > 0:
                logger.info(f"Progress: Indexed {indexed_files} files so far...")

        logger.info(
            f"Directory scan complete: {indexed_files}/{total_files} files indexed in {directory}"
        )
        return indexed_files