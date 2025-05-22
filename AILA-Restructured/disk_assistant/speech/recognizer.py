import os
import json
import time
import threading
import queue
import logging
from tkinter import messagebox

from config import CONFIG

# Try importing Vosk (might not be installed initially)
try:
    from vosk import Model, KaldiRecognizer
    import pyaudio
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

logger = logging.getLogger("DiskAssistant")


class SpeechRecognizer:
    """Handle speech recognition using Vosk."""

    def __init__(self, model_path=None):
        self.model_path = model_path or CONFIG["vosk_model_path"]
        self.recognizer = None
        self.audio = None
        self.stream = None
        self.running = False
        self.queue = queue.Queue()

    def initialize(self):
        """Initialize the speech recognizer."""
        if not VOSK_AVAILABLE:
            logger.warning("Vosk not available, can't initialize speech recognition")
            return False

        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Vosk model not found at {self.model_path}")
                logger.info("Please download a model from https://alphacephei.com/vosk/models")
                messagebox.showwarning(
                    "Vosk Model Missing",
                    f"Speech recognition model not found at {self.model_path}.\n\n"
                    "Please download a model from https://alphacephei.com/vosk/models\n"
                    "and extract it to the specified path.",
                )
                return False

            logger.info(f"Loading Vosk model from {self.model_path}")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
            logger.info("Vosk model loaded successfully")

            try:
                self.audio = pyaudio.PyAudio()
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8000,
                )
                logger.info("Audio input initialized")
                return True
            except Exception as e:
                logger.error(f"Error initializing audio input: {e}")
                messagebox.showerror(
                    "Audio Error",
                    f"Could not initialize microphone: {e}\n\n"
                    "Please check your microphone settings and permissions.",
                )
                return False

        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
            return False

    def start_listening(self):
        """Start listening for speech in a background thread."""
        if not VOSK_AVAILABLE or not self.recognizer:
            return False

        self.running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("Voice recognition started")
        return True

    def stop_listening(self):
        """Stop listening for speech."""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
        logger.info("Voice recognition stopped")

    def _listen_loop(self):
        """Background thread for continuous listening."""
        while self.running:
            try:
                data = self.stream.read(4000, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        logger.info(f"Recognized speech: '{text}'")
                        self.queue.put(text)
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")
                time.sleep(0.5)

    def get_recognized_text(self):
        """Get recognized text from queue if available."""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    @staticmethod
    def is_available():
        """Check if speech recognition is available."""
        return VOSK_AVAILABLE