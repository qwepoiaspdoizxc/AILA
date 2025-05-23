import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

from ai_assistant import AIAssistant
from response_formatter import ResponseFormatter
from config import CONFIG
from utils.logger import logger


class AIChatWidget:
    """AI Chat Widget for the Disk Assistant application."""

    def __init__(
        self, parent, on_search_callback=None, on_file_operation_callback=None
    ):
        self.parent = parent
        self.on_search_callback = on_search_callback
        self.on_file_operation_callback = on_file_operation_callback

        # Initialize AI components
        self.ai_assistant = AIAssistant()
        self.response_formatter = ResponseFormatter()

        # Chat state
        self.conversation_history = []
        self.is_processing = False
        self.response_queue = queue.Queue()

        # UI elements
        self.chat_frame = None
        self.chat_display = None
        self.input_entry = None
        self.send_button = None
        self.clear_button = None
        self.status_label = None
        self.typing_indicator = None

        # Settings
        self.auto_scroll = tk.BooleanVar(value=True)
        self.show_timestamps = tk.BooleanVar(value=True)
        self.response_mode = tk.StringVar(
            value=CONFIG.get("default_response_mode", "text")
        )

        self.setup_ui()
        self.start_response_processor()

        # Initialize AI assistant in background
        self.initialize_ai_async()

    def setup_ui(self):
        """Set up the chat widget UI."""
        # Main frame
        self.chat_frame = ttk.Frame(self.parent)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title and status
        title_frame = ttk.Frame(self.chat_frame)
        title_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(
            title_frame, text="🤖 AI Assistant Chat", font=("Arial", 12, "bold")
        ).pack(side=tk.LEFT)

        self.status_label = ttk.Label(
            title_frame, text="Initializing...", foreground="orange"
        )
        self.status_label.pack(side=tk.RIGHT)

        # Chat display area
        chat_display_frame = ttk.Frame(self.chat_frame)
        chat_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.chat_display = scrolledtext.ScrolledText(
            chat_display_frame,
            wrap=tk.WORD,
            width=60,
            height=20,
            font=("Consolas", 10),
            state=tk.DISABLED,
            bg="#f8f9fa",
            fg="#333333",
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for formatting
        self.setup_text_tags()

        # Input area
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))

        # Input entry
        self.input_entry = ttk.Entry(input_frame, font=("Arial", 10))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.on_send_message)
        self.input_entry.bind("<Shift-Return>", self.on_send_message)

        # Send button
        self.send_button = ttk.Button(
            input_frame, text="Send", command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        # Control buttons
        control_frame = ttk.Frame(self.chat_frame)
        control_frame.pack(fill=tk.X)

        self.clear_button = ttk.Button(
            control_frame, text="Clear Chat", command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        # Settings button
        settings_button = ttk.Button(
            control_frame, text="Settings", command=self.show_settings
        )
        settings_button.pack(side=tk.LEFT, padx=(0, 5))

        # Model info button
        info_button = ttk.Button(
            control_frame, text="Model Info", command=self.show_model_info
        )
        info_button.pack(side=tk.LEFT)

        # Typing indicator
        self.typing_indicator = ttk.Label(control_frame, text="", foreground="gray")
        self.typing_indicator.pack(side=tk.RIGHT)

        # Welcome message
        self.add_system_message(
            "Welcome to AI Disk Assistant! Ask me anything about your files."
        )

    def setup_text_tags(self):
        """Configure text formatting tags."""
        self.chat_display.tag_configure(
            "user", foreground="#0066cc", font=("Arial", 10, "bold")
        )
        self.chat_display.tag_configure("ai", foreground="#009900", font=("Arial", 10))
        self.chat_display.tag_configure(
            "system", foreground="#666666", font=("Arial", 9, "italic")
        )
        self.chat_display.tag_configure(
            "timestamp", foreground="#999999", font=("Arial", 8)
        )
        self.chat_display.tag_configure(
            "error", foreground="#cc0000", font=("Arial", 10)
        )
        self.chat_display.tag_configure("highlight", background="#ffffcc")

    def initialize_ai_async(self):
        """Initialize AI assistant asynchronously."""

        def init_worker():
            try:
                success = self.ai_assistant.initialize()
                if success:
                    self.parent.after(0, lambda: self.update_status("Ready", "green"))
                    self.parent.after(
                        0,
                        lambda: self.add_system_message(
                            "AI assistant initialized successfully. You can now start chatting!"
                        ),
                    )
                else:
                    self.parent.after(
                        0, lambda: self.update_status("AI Unavailable", "red")
                    )
                    self.parent.after(
                        0,
                        lambda: self.add_system_message(
                            "Warning: AI assistant could not be initialized. Check your model configuration."
                        ),
                    )
            except Exception as e:
                logger.error(f"Error initializing AI: {e}")
                self.parent.after(0, lambda: self.update_status("Error", "red"))
                self.parent.after(
                    0,
                    lambda: self.add_system_message(f"Error initializing AI: {str(e)}"),
                )

        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()

    def update_status(self, status: str, color: str = None):
        """Update the status label."""
        self.status_label.config(text=status)
        if color:
            self.status_label.config(foreground=color)

    def on_send_message(self, event=None):
        """Handle Enter key press in input field."""
        if event and event.state & 0x1:  # Shift+Enter = new line
            return
        self.send_message()
        return "break"  # Prevent default behavior

    def send_message(self):
        """Send user message to AI assistant."""
        message = self.input_entry.get().strip()
        if not message or self.is_processing:
            return

        # Clear input
        self.input_entry.delete(0, tk.END)

        # Add user message to chat
        self.add_user_message(message)

        # Process message in background
        self.process_message_async(message)

    def process_message_async(self, message: str):
        """Process message asynchronously."""
        if not self.ai_assistant.is_available():
            self.add_error_message(
                "AI assistant is not available. Please check the configuration."
            )
            return

        self.is_processing = True
        self.send_button.config(state=tk.DISABLED)
        self.show_typing_indicator()

        def process_worker():
            try:
                # Analyze intent first
                intent_analysis = self.ai_assistant.analyze_query_intent(message)

                # Prepare context based on intent
                context = self.prepare_context(intent_analysis, message)

                # Generate AI response
                ai_response = self.ai_assistant.generate_response(message, context)

                # Format response based on intent
                formatted_response = self.format_response_by_intent(
                    ai_response, intent_analysis, context
                )

                # Queue response for UI thread
                self.response_queue.put(
                    {
                        "type": "ai_response",
                        "content": formatted_response,
                        "intent": intent_analysis,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.response_queue.put(
                    {
                        "type": "error",
                        "content": f"Error processing your request: {str(e)}",
                    }
                )
            finally:
                self.response_queue.put({"type": "processing_complete"})

        thread = threading.Thread(target=process_worker, daemon=True)
        thread.start()

    def prepare_context(self, intent_analysis: Dict, message: str) -> Dict:
        """Prepare context based on intent analysis."""
        context = {
            "user_intent": intent_analysis.get("intent", "general"),
            "confidence": intent_analysis.get("confidence", 0.5),
        }

        # Add search results if this is a search intent
        if intent_analysis.get("intent") == "search" and self.on_search_callback:
            try:
                search_results = self.on_search_callback(message)
                context["search_results"] = search_results[:10]  # Limit results
            except Exception as e:
                logger.error(f"Error getting search results: {e}")

        return context

    def format_response_by_intent(
        self, ai_response: str, intent_analysis: Dict, context: Dict
    ) -> str:
        """Format response based on detected intent."""
        intent = intent_analysis.get("intent", "general")

        if intent == "search" and "search_results" in context:
            return self.response_formatter.format_search_response(
                ai_response=ai_response,
                results=context["search_results"],
                query=intent_analysis.get("parameters", {}).get("query", ""),
                response_mode=self.response_mode.get(),
            )
        elif intent == "analyze":
            # Mock analysis result for now
            analysis_result = {
                "insights": ["Data analysis would be performed here"],
                "recommendations": [
                    "Check data quality",
                    "Consider data visualization",
                ],
            }
            return self.response_formatter.format_data_analysis_response(
                ai_response=ai_response,
                analysis_result=analysis_result,
                response_mode=self.response_mode.get(),
            )
        elif intent == "help":
            return self.response_formatter.format_help_response(
                ai_response=ai_response, response_mode=self.response_mode.get()
            )
        else:
            return self.response_formatter.format_conversation_response(
                ai_response=ai_response, response_mode=self.response_mode.get()
            )

    def start_response_processor(self):
        """Start the response processor thread."""

        def process_responses():
            while True:
                try:
                    response = self.response_queue.get(timeout=0.1)

                    if response["type"] == "ai_response":
                        self.parent.after(
                            0, lambda r=response: self.add_ai_message(r["content"])
                        )
                    elif response["type"] == "error":
                        self.parent.after(
                            0, lambda r=response: self.add_error_message(r["content"])
                        )
                    elif response["type"] == "processing_complete":
                        self.parent.after(0, self.hide_typing_indicator)
                        self.parent.after(
                            0, lambda: setattr(self, "is_processing", False)
                        )
                        self.parent.after(
                            0, lambda: self.send_button.config(state=tk.NORMAL)
                        )

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing response: {e}")

        thread = threading.Thread(target=process_responses, daemon=True)
        thread.start()

    def add_user_message(self, message: str):
        """Add user message to chat display."""
        self.add_message("You", message, "user")

    def add_ai_message(self, message: str):
        """Add AI message to chat display."""
        # Handle both text and voice-enabled responses
        if isinstance(message, dict) and "text" in message:
            display_message = message["text"]
        else:
            display_message = message

        self.add_message("AI Assistant", display_message, "ai")

    def add_system_message(self, message: str):
        """Add system message to chat display."""
        self.add_message("System", message, "system")

    def add_error_message(self, message: str):
        """Add error message to chat display."""
        self.add_message("Error", message, "error")

    def add_message(self, sender: str, message: str, tag: str):
        """Add a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)

        # Add timestamp if enabled
        timestamp = ""
        if self.show_timestamps.get():
            timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] "

        # Add sender and message
        if timestamp:
            self.chat_display.insert(tk.END, timestamp, "timestamp")

        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n\n")

        self.chat_display.config(state=tk.DISABLED)

        # Auto-scroll if enabled
        if self.auto_scroll.get():
            self.chat_display.see(tk.END)

        # Store in conversation history
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "sender": sender,
                "message": message,
                "tag": tag,
            }
        )

    def show_typing_indicator(self):
        """Show typing indicator."""
        self.typing_indicator.config(text="🤖 AI is thinking...")

    def hide_typing_indicator(self):
        """Hide typing indicator."""
        self.typing_indicator.config(text="")

    def clear_chat(self):
        """Clear the chat display and history."""
        if messagebox.askyesno(
            "Clear Chat", "Are you sure you want to clear the chat history?"
        ):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)

            self.conversation_history.clear()
            self.ai_assistant.clear_history()

            self.add_system_message("Chat cleared. How can I help you?")

    def show_settings(self):
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.parent)
        settings_window.title("Chat Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)

        # Response mode
        ttk.Label(settings_window, text="Response Mode:").pack(pady=5)
        mode_frame = ttk.Frame(settings_window)
        mode_frame.pack(pady=5)

        ttk.Radiobutton(
            mode_frame, text="Text Only", variable=self.response_mode, value="text"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Voice Only", variable=self.response_mode, value="voice"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Both", variable=self.response_mode, value="both"
        ).pack(side=tk.LEFT, padx=5)

        # Display options
        ttk.Separator(settings_window).pack(fill=tk.X, pady=10)

        ttk.Checkbutton(
            settings_window, text="Show Timestamps", variable=self.show_timestamps
        ).pack(pady=2)
        ttk.Checkbutton(
            settings_window, text="Auto-scroll", variable=self.auto_scroll
        ).pack(pady=2)

        # AI settings
        ttk.Separator(settings_window).pack(fill=tk.X, pady=10)
        ttk.Label(
            settings_window, text="AI Settings:", font=("Arial", 10, "bold")
        ).pack(pady=5)

        # Temperature
        temp_frame = ttk.Frame(settings_window)
        temp_frame.pack(pady=5, fill=tk.X, padx=20)
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT)
        temp_var = tk.DoubleVar(value=self.ai_assistant.temperature)
        temp_scale = ttk.Scale(
            temp_frame, from_=0.1, to=1.0, variable=temp_var, orient=tk.HORIZONTAL
        )
        temp_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Max tokens
        tokens_frame = ttk.Frame(settings_window)
        tokens_frame.pack(pady=5, fill=tk.X, padx=20)
        ttk.Label(tokens_frame, text="Max Tokens:").pack(side=tk.LEFT)
        tokens_var = tk.IntVar(value=self.ai_assistant.max_tokens)
        ttk.Spinbox(tokens_frame, from_=100, to=2048, textvariable=tokens_var).pack(
            side=tk.RIGHT
        )

        # Buttons
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(pady=20)

        def apply_settings():
            self.ai_assistant.temperature = temp_var.get()
            self.ai_assistant.max_tokens = tokens_var.get()
            settings_window.destroy()
            self.add_system_message("Settings updated successfully.")

        ttk.Button(button_frame, text="Apply", command=apply_settings).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(
            side=tk.LEFT, padx=5
        )

    def show_model_info(self):
        """Show AI model information."""
        info = self.ai_assistant.get_model_info()

        info_window = tk.Toplevel(self.parent)
        info_window.title("AI Model Information")
        info_window.geometry("500x400")

        info_text = scrolledtext.ScrolledText(
            info_window, wrap=tk.WORD, font=("Consolas", 10)
        )
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Format model info
        formatted_info = "🤖 AI Model Information\n" + "=" * 50 + "\n\n"
        for key, value in info.items():
            formatted_info += f"{key.replace('_', ' ').title()}: {value}\n"

        info_text.insert(tk.END, formatted_info)
        info_text.config(state=tk.DISABLED)

        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)

    def export_chat(self, filename: Optional[str] = None):
        """Export chat history to file."""
        if not filename:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ],
            )

        if not filename:
            return

        try:
            if filename.endswith(".json"):
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(
                        self.conversation_history, f, indent=2, ensure_ascii=False
                    )
            else:
                with open(filename, "w", encoding="utf-8") as f:
                    for entry in self.conversation_history:
                        timestamp = entry["timestamp"]
                        sender = entry["sender"]
                        message = entry["message"]
                        f.write(f"[{timestamp}] {sender}: {message}\n\n")

            self.add_system_message(f"Chat history exported to {filename}")

        except Exception as e:
            logger.error(f"Error exporting chat: {e}")
            self.add_error_message(f"Failed to export chat: {str(e)}")

    def get_widget(self) -> ttk.Frame:
        """Get the main widget frame."""
        return self.chat_frame

    def focus_input(self):
        """Focus the input entry field."""
        self.input_entry.focus_set()

    def insert_text(self, text: str):
        """Insert text into the input field."""
        current = self.input_entry.get()
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, current + text)

    def set_search_callback(self, callback):
        """Set the search callback function."""
        self.on_search_callback = callback

    def set_file_operation_callback(self, callback):
        """Set the file operation callback function."""
        self.on_file_operation_callback = callback

    def destroy(self):
        """Clean up resources."""
        try:
            # Clear queues
            while not self.response_queue.empty():
                self.response_queue.get_nowait()

            # Clear AI assistant
            if hasattr(self.ai_assistant, "model") and self.ai_assistant.model:
                self.ai_assistant.model = None

            logger.info("AI Chat Widget destroyed successfully")
        except Exception as e:
            logger.error(f"Error destroying AI Chat Widget: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Create test window
    root = tk.Tk()
    root.title("AI Chat Widget Test")
    root.geometry("800x600")

    # Mock search callback
    def mock_search(query):
        return [
            {
                "filename": "test1.txt",
                "path": "/home/user/test1.txt",
                "file_type": "text",
                "last_modified": "2024-01-15 10:30:00",
            },
            {
                "filename": "test2.pdf",
                "path": "/home/user/documents/test2.pdf",
                "file_type": "pdf",
                "last_modified": "2024-01-14 15:45:00",
            },
        ]

    # Create chat widget
    chat_widget = AIChatWidget(root, on_search_callback=mock_search)

    # Add test menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    chat_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Chat", menu=chat_menu)
    chat_menu.add_command(label="Export Chat", command=chat_widget.export_chat)
    chat_menu.add_command(label="Clear Chat", command=chat_widget.clear_chat)
    chat_menu.add_separator()
    chat_menu.add_command(label="Settings", command=chat_widget.show_settings)
    chat_menu.add_command(label="Model Info", command=chat_widget.show_model_info)

    def on_closing():
        chat_widget.destroy()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_closing()
