import os
import time
import threading
import queue
import logging

from config import CONFIG

# Third-party imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    raise ImportError("watchdog module not found. Please install it with: pip install watchdog")

logger = logging.getLogger("DiskAssistant")


class FileSystemMonitor(FileSystemEventHandler):
    """Monitor file system changes and update index accordingly."""

    def __init__(self, file_index):
        self.file_index = file_index
        self.observer = Observer()
        self.pending_events = queue.Queue()
        self.event_processor_running = False
        self.lock = threading.Lock()

    def start_monitoring(self, directories=None):
        """Start monitoring specified directories."""
        if directories is None:
            directories = CONFIG["directories_to_watch"]

        for directory in directories:
            if os.path.exists(directory) and os.path.isdir(directory):
                try:
                    self.observer.schedule(self, directory, recursive=True)
                    logger.info(f"Monitoring directory: {directory}")
                except Exception as e:
                    logger.error(f"Error monitoring directory {directory}: {e}")

        try:
            self.observer.start()
            # Start event processing thread
            threading.Thread(target=self._process_events, daemon=True).start()
        except Exception as e:
            logger.error(f"Error starting file system monitor: {e}")

    def stop_monitoring(self):
        """Stop the file system monitor."""
        try:
            self.observer.stop()
            self.observer.join()
            logger.info("File system monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping file system monitor: {e}")

    def on_created(self, event):
        """Handle file creation events by queueing them."""
        if not event.is_directory:
            self.pending_events.put(("create", event.src_path))

    def on_modified(self, event):
        """Handle file modification events by queueing them."""
        if not event.is_directory:
            self.pending_events.put(("modify", event.src_path))

    def on_deleted(self, event):
        """Handle file deletion events by queueing them."""
        if not event.is_directory:
            self.pending_events.put(("delete", event.src_path))

    def on_moved(self, event):
        """Handle file move/rename events by queueing them."""
        if not event.is_directory:
            self.pending_events.put(("delete", event.src_path))
            self.pending_events.put(("create", event.dest_path))

    def _process_events(self):
        """Background thread to process file events with rate limiting."""
        self.event_processor_running = True
        event_cache = {}  # To eliminate duplicate events

        while self.event_processor_running:
            try:
                # Process up to 10 events at a time
                for _ in range(10):
                    try:
                        event_type, path = self.pending_events.get_nowait()

                        # Skip if we've already processed this event recently
                        event_key = f"{event_type}:{path}"
                        if event_key in event_cache:
                            continue

                        # Process the event
                        if event_type == "create" or event_type == "modify":
                            self.file_index.add_file(path)
                        elif event_type == "delete":
                            self.file_index.remove_file(path)

                        # Add to cache to prevent duplicate processing
                        event_cache[event_key] = time.time()

                    except queue.Empty:
                        break

                # Clean old cache entries
                now = time.time()
                for key in list(event_cache.keys()):
                    if now - event_cache[key] > 10:  # Remove after 10 seconds
                        del event_cache[key]

                # Sleep to rate limit
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                time.sleep(1)  # Wait a bit longer if there's an error