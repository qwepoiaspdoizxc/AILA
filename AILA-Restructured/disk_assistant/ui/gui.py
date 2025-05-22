import os
import sys
import json
import time
import threading
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

from config import CONFIG
from utils.logger import logger
from indexer.file_index import FileIndex
from indexer.monitor import FileSystemMonitor
from nlp.query_processor import QueryProcessor
from speech.recognizer import SpeechRecognizer


class DiskAssistantGUI:
    """Main GUI for the Disk Assistant"""

    def __init__(self, root):
        self.root = root
        self.root.title("Local Disk Assistant")
        self.root.geometry("800x600")

        # Initialize components
        self.file_index = FileIndex()
        self.file_monitor = FileSystemMonitor(self.file_index)
        self.query_processor = QueryProcessor()
        
        # Initialize speech recognizer only if available
        try:
            self.speech_recognizer = SpeechRecognizer()
        except ImportError:
            self.speech_recognizer = None
            logger.warning("Speech recognition not available")

        # Status variables
        self.status_var = tk.StringVar(value="Ready")
        self.listening_var = tk.BooleanVar(value=False)
        self.indexing_in_progress = False

        # Create UI
        self._create_ui()

        # Start background processes
        self._start_background_processes()

    def _create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Add Directory to Index...", command=self._add_directory
        )
        file_menu.add_command(label="Reindex All", command=self._trigger_reindex)
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self._show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="View Log", command=self._view_log)

        # Top bar with search
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        # Microphone button (if speech recognition available)
        if self.speech_recognizer:
            self.mic_btn = ttk.Button(
                search_frame, text="🎤", width=3, command=self._toggle_listening
            )
            self.mic_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(
            search_frame, textvariable=self.search_var, width=40
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry.bind("<Return>", self._perform_search)

        # Search button
        search_btn = ttk.Button(
            search_frame, text="Search", command=self._perform_search
        )
        search_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results list with scrollbar
        results_scroll = ttk.Scrollbar(results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_list = ttk.Treeview(
            results_frame, columns=("Path", "Type", "Modified"), show="headings"
        )

        # Configure columns
        self.results_list.heading("Path", text="File Path")
        self.results_list.heading("Type", text="Type")
        self.results_list.heading("Modified", text="Modified")
        
        self.results_list.column("Path", width=400)
        self.results_list.column("Type", width=100)
        self.results_list.column("Modified", width=150)

        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.results_list.yview)
        self.results_list.config(yscrollcommand=results_scroll.set)

        # Bind double-click to open file
        self.results_list.bind("<Double-1>", self._open_selected_file)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        # Status label
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)

        # Voice status indicator
        if self.speech_recognizer:
            voice_frame = ttk.Frame(status_frame)
            voice_frame.pack(side=tk.RIGHT)

            self.voice_indicator = ttk.Label(voice_frame, text="🎤 Voice:")
            self.voice_indicator.pack(side=tk.LEFT, padx=(0, 5))

            self.voice_status = ttk.Label(voice_frame, text="Off")
            self.voice_status.pack(side=tk.LEFT)

    def _start_background_processes(self):
        """Start background monitoring and periodic scans"""
        # Start file monitoring
        self.file_monitor.start_monitoring()

        # Schedule initial scan if database is empty
        self.root.after(1000, self._check_initial_scan)

        # Set up speech recognition if available
        if self.speech_recognizer:
            if self.speech_recognizer.initialize():
                # Check for new voice input periodically
                self.root.after(500, self._check_voice_input)

        # Schedule periodic rescans
        self.root.after(CONFIG["scan_interval"] * 1000, self._periodic_rescan)

    def _check_initial_scan(self):
        """Check if initial scan is needed and perform it"""
        try:
            conn = sqlite3.connect(CONFIG["db_path"])
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM files")
            count = cursor.fetchone()[0]
            conn.close()

            if count == 0:
                # No files indexed yet, perform initial scan
                self._trigger_reindex()
            else:
                self.status_var.set(f"Ready - {count} files indexed")
        except sqlite3.Error as e:
            logger.error(f"Error checking database: {e}")
            self.status_var.set("Error checking database")

    def _periodic_rescan(self):
        """Perform periodic rescan of watched directories"""
        if not self.indexing_in_progress:
            threading.Thread(target=self._background_rescan, daemon=True).start()

        # Schedule next scan
        self.root.after(CONFIG["scan_interval"] * 1000, self._periodic_rescan)

    def _background_rescan(self):
        """Background thread for rescanning directories"""
        self.indexing_in_progress = True
        self.status_var.set("Performing scheduled rescan...")

        total_indexed = 0
        for directory in CONFIG["directories_to_watch"]:
            total_indexed += self.file_index.scan_directory(directory)

        self.status_var.set(f"Rescan complete - {total_indexed} files updated")
        self.indexing_in_progress = False

    def _trigger_reindex(self):
        """Start a full reindexing operation"""
        if self.indexing_in_progress:
            messagebox.showinfo("Indexing", "Indexing is already in progress")
            return

        confirm = messagebox.askyesno(
            "Confirm Reindex", "This will scan all watched directories. Continue?"
        )

        if confirm:
            threading.Thread(target=self._background_reindex, daemon=True).start()

    def _background_reindex(self):
        """Background thread for full reindexing"""
        self.indexing_in_progress = True
        self.status_var.set("Indexing in progress...")

        total_indexed = 0
        for directory in CONFIG["directories_to_watch"]:
            total_indexed += self.file_index.scan_directory(directory)

        self.status_var.set(f"Indexing complete - {total_indexed} files indexed")
        self.indexing_in_progress = False

    def _perform_search(self, event=None):
        """Execute a search based on user input"""
        query = self.search_var.get().strip()
        if not query:
            return

        # Process the query
        processed = self.query_processor.process_query(query)
        search_query = self.query_processor.generate_search_query(processed)

        # Update status
        self.status_var.set(f"Searching for: {search_query}")

        # Clear previous results
        for item in self.results_list.get_children():
            self.results_list.delete(item)

        # Perform the search
        results = self.file_index.search(search_query)

        # Display results
        for result in results:
            path = result["path"]
            filename = os.path.basename(path)
            file_type = result["file_type"]
            modified = result["last_modified"].strftime("%Y-%m-%d %H:%M")

            self.results_list.insert("", tk.END, values=(path, file_type, modified))

        # Update status
        self.status_var.set(f"Found {len(results)} results")

    def _open_selected_file(self, event=None):
        """Open the selected file with default application"""
        selection = self.results_list.selection()
        if not selection:
            return

        item = selection[0]
        file_path = self.results_list.item(item, "values")[0]

        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":  # macOS
                import subprocess
                subprocess.call(["open", file_path])
            else:  # Linux
                import subprocess
                subprocess.call(["xdg-open", file_path])

            logger.info(f"Opened file: {file_path}")
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            messagebox.showerror("Error", f"Could not open file: {str(e)}")

    def _toggle_listening(self):
        """Toggle speech recognition on/off"""
        if not self.speech_recognizer:
            return

        if self.listening_var.get():
            # Stop listening
            self.speech_recognizer.stop_listening()
            self.listening_var.set(False)
            self.voice_status.config(text="Off")
            self.mic_btn.config(text="🎤")
        else:
            # Start listening
            if self.speech_recognizer.start_listening():
                self.listening_var.set(True)
                self.voice_status.config(text="On")
                self.mic_btn.config(text="🔴")

    def _check_voice_input(self):
        """Check for new voice input"""
        if self.speech_recognizer and self.listening_var.get():
            text = self.speech_recognizer.get_recognized_text()
            if text:
                # Set the recognized text to search box
                self.search_var.set(text)
                # Automatically perform search
                self._perform_search()

        # Continue checking
        self.root.after(500, self._check_voice_input)

    def _add_directory(self):
        """Add a new directory to watch"""
        directory = filedialog.askdirectory(title="Select Directory to Index")
        if directory:
            # Add to config
            if directory not in CONFIG["directories_to_watch"]:
                CONFIG["directories_to_watch"].append(directory)

                # Start monitoring the new directory
                self.file_monitor.start_monitoring([directory])

                # Trigger indexing for the new directory
                threading.Thread(
                    target=lambda: self.file_index.scan_directory(directory),
                    daemon=True,
                ).start()

                self.status_var.set(f"Added directory: {directory}")

    def _show_settings(self):
        """Show settings dialog"""
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings")
        settings_win.geometry("500x400")
        settings_win.transient(self.root)
        settings_win.grab_set()

        # Create settings UI
        frame = ttk.Frame(settings_win, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Directories section
        ttk.Label(frame, text="Watched Directories:").pack(anchor=tk.W)

        # Listbox for directories with scrollbar
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        dir_scroll = ttk.Scrollbar(dir_frame)
        dir_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        dir_listbox = tk.Listbox(dir_frame)
        dir_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        dir_scroll.config(command=dir_listbox.yview)
        dir_listbox.config(yscrollcommand=dir_scroll.set)

        # Populate listbox
        for directory in CONFIG["directories_to_watch"]:
            dir_listbox.insert(tk.END, directory)

        # Buttons for directory management
        dir_buttons = ttk.Frame(frame)
        dir_buttons.pack(fill=tk.X, pady=(0, 10))

        add_btn = ttk.Button(
            dir_buttons, text="Add", command=lambda: self._settings_add_dir(dir_listbox)
        )
        add_btn.pack(side=tk.LEFT, padx=(0, 5))

        remove_btn = ttk.Button(
            dir_buttons,
            text="Remove",
            command=lambda: self._settings_remove_dir(dir_listbox),
        )
        remove_btn.pack(side=tk.LEFT)

        # File extensions
        ttk.Label(frame, text="Indexed File Extensions:").pack(anchor=tk.W)
        ext_var = tk.StringVar(value=", ".join(CONFIG["file_extensions"]))
        ext_entry = ttk.Entry(frame, textvariable=ext_var)
        ext_entry.pack(fill=tk.X, pady=(5, 10))

        # Max file size
        size_frame = ttk.Frame(frame)
        size_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(size_frame, text="Maximum File Size (MB):").pack(side=tk.LEFT)

        size_var = tk.IntVar(value=CONFIG["max_file_size"] // (1024 * 1024))
        size_entry = ttk.Spinbox(
            size_frame, from_=1, to=100, textvariable=size_var, width=5
        )
        size_entry.pack(side=tk.LEFT, padx=(5, 0))

        # Save button
        save_btn = ttk.Button(
            frame,
            text="Save Settings",
            command=lambda: self._save_settings(
                dir_listbox.get(0, tk.END), ext_var.get(), size_var.get(), settings_win
            ),
        )
        save_btn.pack(side=tk.RIGHT, pady=(10, 0))

    def _settings_add_dir(self, listbox):
        """Add directory from settings dialog"""
        directory = filedialog.askdirectory(title="Select Directory to Watch")
        if directory and directory not in listbox.get(0, tk.END):
            listbox.insert(tk.END, directory)

    def _settings_remove_dir(self, listbox):
        """Remove directory from settings dialog"""
        selection = listbox.curselection()
        if selection:
            listbox.delete(selection[0])

    def _save_settings(self, directories, extensions, max_size, window):
        """Save settings and close dialog"""
        # Update config
        CONFIG["directories_to_watch"] = list(directories)
        CONFIG["file_extensions"] = [ext.strip() for ext in extensions.split(",")]
        CONFIG["max_file_size"] = max_size * 1024 * 1024  # Convert MB to bytes

        # Restart monitoring with new directories
        self.file_monitor.stop_monitoring()
        self.file_monitor.start_monitoring()

        # Save config to file
        try:
            with open("disk_assistant_config.json", "w") as f:
                json.dump(CONFIG, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            messagebox.showerror("Error", f"Could not save config: {str(e)}")

        window.destroy()
        self.status_var.set("Settings updated")

    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Disk Assistant",
            "Disk Assistant v1.0\n\n"
            "A local file indexing and searching tool.\n"
            "© 2025 Local Disk Assistant Project\n\n"
            "Uses: Python, tkinter, Whoosh, spaCy, Vosk",
        )

    def _view_log(self):
        """Open log file in default text editor"""
        log_path = "disk_assistant.log"
        if os.path.exists(log_path):
            try:
                if sys.platform == "win32":
                    os.startfile(log_path)
                elif sys.platform == "darwin":  # macOS
                    import subprocess
                    subprocess.call(["open", log_path])
                else:  # Linux
                    import subprocess
                    subprocess.call(["xdg-open", log_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open log file: {str(e)}")
        else:
            messagebox.showinfo("Log", "Log file does not exist yet")

    def on_close(self):
        """Handle application closing"""
        # Stop background processes
        if self.file_monitor:
            self.file_monitor.stop_monitoring()

        if self.speech_recognizer and self.listening_var.get():
            self.speech_recognizer.stop_listening()

        # Close the application
        self.root.destroy()