import os
import json
import time
import threading
import queue
import tempfile
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
import logging

# Audio processing imports
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import vosk

    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from config import CONFIG
from utils.logger import get_logger
from audio_utils import AudioRecorder

logger = get_logger("WakeWordDetector")


class WakeWordDetector:
    """Advanced wake word detection using Vosk speech recognition."""

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.is_listening = False
        self.is_activated = False
        self.model = None
        self.recognizer = None
        self.audio_recorder = None
        self.audio_stream = None
        self.detection_thread = None
        self.audio_queue = queue.Queue()

        # Wake word configuration
        self.wake_words = self.config.get("wake_words", ["hey assistant"])
        self.activation_threshold = self.config.get("activation_threshold", 0.7)
        self.voice_timeout = self.config.get("voice_timeout", 5.0)

        # Callbacks
        self.wake_word_callback = None
        self.command_callback = None
        self.timeout_callback = None

        # Audio settings
        self.sample_rate = self.config.get(
            "sample_rate", 16000
        )  # Vosk works best with 16kHz
        self.channels = 1  # Vosk requires mono audio
        self.chunk_size = 4000  # Audio chunk size for processing

        # State tracking
        self.last_activation_time = 0
        self.command_buffer = []
        self.confidence_scores = []

    def initialize(self) -> bool:
        """Initialize the wake word detector."""
        if not self._check_dependencies():
            return False

        if not self._load_vosk_model():
            return False

        if not self._initialize_audio():
            return False

        logger.info("Wake word detector initialized successfully")
        return True

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        missing_deps = []

        if not PYAUDIO_AVAILABLE:
            missing_deps.append("pyaudio")
        if not VOSK_AVAILABLE:
            missing_deps.append("vosk")
        if not NUMPY_AVAILABLE:
            missing_deps.append("numpy")

        if missing_deps:
            logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
            return False

        return True

    def _load_vosk_model(self) -> bool:
        """Load the Vosk speech recognition model."""
        model_path = self.config.get("vosk_model_path")

        if not model_path or not os.path.exists(model_path):
            logger.error(f"Vosk model not found at: {model_path}")
            return False

        try:
            # Set Vosk log level to reduce noise
            vosk.SetLogLevel(-1)

            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

            # Configure recognizer for partial results
            self.recognizer.SetWords(True)
            self.recognizer.SetPartialWords(True)

            logger.info(f"Loaded Vosk model from: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            return False

    def _initialize_audio(self) -> bool:
        """Initialize audio recording for wake word detection."""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available for audio recording")
            return False

        try:
            self.audio = pyaudio.PyAudio()

            # Find the best input device
            device_index = self._find_best_input_device()

            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
            )

            logger.info(f"Audio initialized: {self.sample_rate}Hz, {self.channels}ch")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            return False

    def _find_best_input_device(self) -> Optional[int]:
        """Find the best available input device."""
        try:
            default_device = self.audio.get_default_input_device_info()
            logger.info(f"Using default input device: {default_device['name']}")
            return default_device["index"]

        except Exception as e:
            logger.warning(f"Could not get default input device: {e}")

            # Try to find any available input device
            try:
                for i in range(self.audio.get_device_count()):
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info["maxInputChannels"] > 0:
                        logger.info(f"Using input device: {device_info['name']}")
                        return i
            except Exception as e2:
                logger.error(f"Could not find any input device: {e2}")

        return None

    def set_wake_word_callback(self, callback: Callable[[str, float], None]):
        """Set callback for wake word detection."""
        self.wake_word_callback = callback

    def set_command_callback(self, callback: Callable[[str, float], None]):
        """Set callback for command recognition after wake word."""
        self.command_callback = callback

    def set_timeout_callback(self, callback: Callable[[], None]):
        """Set callback for command timeout."""
        self.timeout_callback = callback

    def start_listening(self) -> bool:
        """Start continuous wake word detection."""
        if self.is_listening:
            logger.warning("Already listening for wake words")
            return False

        if not self.model or not self.audio_stream:
            logger.error("Wake word detector not properly initialized")
            return False

        self.is_listening = True
        self.is_activated = False

        # Start detection thread
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.detection_thread.start()

        logger.info("Started listening for wake words")
        return True

    def stop_listening(self) -> bool:
        """Stop wake word detection."""
        if not self.is_listening:
            return True

        self.is_listening = False
        self.is_activated = False

        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)

        logger.info("Stopped listening for wake words")
        return True

    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        logger.info("Wake word detection loop started")

        while self.is_listening:
            try:
                # Read audio data
                data = self.audio_stream.read(
                    self.chunk_size, exception_on_overflow=False
                )

                # Process audio with Vosk
                if self.recognizer.AcceptWaveform(data):
                    # Complete recognition result
                    result = json.loads(self.recognizer.Result())
                    self._process_recognition_result(result, final=True)
                else:
                    # Partial recognition result
                    partial = json.loads(self.recognizer.PartialResult())
                    self._process_recognition_result(partial, final=False)

            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                break

        logger.info("Wake word detection loop ended")

    def _process_recognition_result(self, result: Dict, final: bool = False):
        """Process recognition result from Vosk."""
        text_key = "text" if final else "partial"

        if text_key not in result or not result[text_key]:
            return

        recognized_text = result[text_key].lower().strip()

        if not recognized_text:
            return

        # Get confidence score if available
        confidence = self._extract_confidence(result)

        if self.is_activated:
            # We're in command mode - capture the command
            self._process_command(recognized_text, confidence, final)
        else:
            # Check for wake words
            self._check_wake_words(recognized_text, confidence)

    def _extract_confidence(self, result: Dict) -> float:
        """Extract confidence score from Vosk result."""
        if "result" in result and result["result"]:
            # Average confidence of all words
            confidences = [word.get("conf", 0.0) for word in result["result"]]
            return sum(confidences) / len(confidences) if confidences else 0.0
        return 0.5  # Default confidence for partial results

    def _check_wake_words(self, text: str, confidence: float):
        """Check if recognized text contains wake words."""
        for wake_word in self.wake_words:
            wake_word_lower = wake_word.lower()

            # Check for exact match or partial match
            if wake_word_lower in text:
                # Calculate similarity score
                similarity = self._calculate_similarity(wake_word_lower, text)

                if similarity >= self.activation_threshold:
                    self._activate_wake_word(wake_word, confidence * similarity)
                    break

    def _calculate_similarity(self, wake_word: str, text: str) -> float:
        """Calculate similarity between wake word and recognized text."""
        # Simple word-based similarity
        wake_words = set(wake_word.split())
        text_words = set(text.split())

        if not wake_words:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(wake_words.intersection(text_words))
        union = len(wake_words.union(text_words))

        jaccard_sim = intersection / union if union > 0 else 0.0

        # Boost score if wake word appears as substring
        substring_boost = 0.3 if wake_word in text else 0.0

        return min(1.0, jaccard_sim + substring_boost)

    def _activate_wake_word(self, wake_word: str, confidence: float):
        """Handle wake word activation."""
        self.is_activated = True
        self.last_activation_time = time.time()
        self.command_buffer = []

        logger.info(f"Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")

        # Call wake word callback
        if self.wake_word_callback:
            try:
                self.wake_word_callback(wake_word, confidence)
            except Exception as e:
                logger.error(f"Error in wake word callback: {e}")

        # Start command timeout timer
        threading.Timer(self.voice_timeout, self._handle_command_timeout).start()

    def _process_command(self, text: str, confidence: float, final: bool):
        """Process command after wake word activation."""
        current_time = time.time()

        # Check if we've timed out
        if current_time - self.last_activation_time > self.voice_timeout:
            self._deactivate()
            return

        # Add to command buffer
        self.command_buffer.append(
            {
                "text": text,
                "confidence": confidence,
                "timestamp": current_time,
                "final": final,
            }
        )

        # If this is a final result, process the complete command
        if final and text.strip():
            self._handle_complete_command(text, confidence)

    def _handle_complete_command(self, command_text: str, confidence: float):
        """Handle a complete command."""
        logger.info(
            f"Command recognized: '{command_text}' (confidence: {confidence:.2f})"
        )

        # Call command callback
        if self.command_callback:
            try:
                self.command_callback(command_text, confidence)
            except Exception as e:
                logger.error(f"Error in command callback: {e}")

        # Deactivate after processing command
        self._deactivate()

    def _handle_command_timeout(self):
        """Handle timeout waiting for command."""
        if self.is_activated:
            logger.info("Command timeout - deactivating")

            if self.timeout_callback:
                try:
                    self.timeout_callback()
                except Exception as e:
                    logger.error(f"Error in timeout callback: {e}")

            self._deactivate()

    def _deactivate(self):
        """Deactivate command mode and return to wake word detection."""
        self.is_activated = False
        self.command_buffer = []
        logger.debug("Deactivated - listening for wake words")

    def add_wake_word(self, wake_word: str):
        """Add a new wake word to the detection list."""
        if wake_word.lower() not in [w.lower() for w in self.wake_words]:
            self.wake_words.append(wake_word.lower())
            logger.info(f"Added wake word: '{wake_word}'")

    def remove_wake_word(self, wake_word: str):
        """Remove a wake word from the detection list."""
        wake_word_lower = wake_word.lower()
        self.wake_words = [w for w in self.wake_words if w.lower() != wake_word_lower]
        logger.info(f"Removed wake word: '{wake_word}'")

    def set_activation_threshold(self, threshold: float):
        """Set the activation threshold for wake word detection."""
        self.activation_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Set activation threshold to: {self.activation_threshold}")

    def set_voice_timeout(self, timeout: float):
        """Set the timeout for voice commands after wake word."""
        self.voice_timeout = max(1.0, timeout)
        logger.info(f"Set voice timeout to: {self.voice_timeout} seconds")

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        return {
            "is_listening": self.is_listening,
            "is_activated": self.is_activated,
            "wake_words": self.wake_words,
            "activation_threshold": self.activation_threshold,
            "voice_timeout": self.voice_timeout,
            "model_loaded": self.model is not None,
            "audio_initialized": self.audio_stream is not None,
            "last_activation": self.last_activation_time,
        }

    def get_audio_devices(self) -> List[Dict]:
        """Get list of available audio input devices."""
        devices = []

        if not self.audio:
            return devices

        try:
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    devices.append(
                        {
                            "index": i,
                            "name": device_info["name"],
                            "channels": device_info["maxInputChannels"],
                            "sample_rate": int(device_info["defaultSampleRate"]),
                        }
                    )
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")

        return devices

    def test_recognition(self, duration: float = 5.0) -> List[str]:
        """Test speech recognition for a specified duration."""
        if not self.is_listening:
            logger.error("Detector not listening - start listening first")
            return []

        logger.info(f"Testing recognition for {duration} seconds...")

        results = []
        start_time = time.time()

        # Temporarily store original callback
        original_callback = self.command_callback

        # Set test callback
        def test_callback(text, confidence):
            results.append(f"{text} (confidence: {confidence:.2f})")

        self.command_callback = test_callback

        # Force activation for testing
        self.is_activated = True
        self.last_activation_time = start_time

        # Wait for test duration
        time.sleep(duration)

        # Restore original state
        self.command_callback = original_callback
        self.is_activated = False

        logger.info(f"Recognition test completed. Results: {len(results)}")
        return results

    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()

        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            self.audio_stream = None

        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating audio: {e}")
            self.audio = None

        logger.info("Wake word detector cleaned up")


# Utility functions
def create_wake_word_detector(config: Dict = None) -> WakeWordDetector:
    """Create and return a WakeWordDetector instance."""
    return WakeWordDetector(config)


def test_wake_word_detection(duration: float = 10.0, config: Dict = None) -> bool:
    """Test wake word detection functionality."""
    detector = WakeWordDetector(config)

    def on_wake_word(word, confidence):
        print(f"Wake word detected: '{word}' (confidence: {confidence:.2f})")

    def on_command(command, confidence):
        print(f"Command: '{command}' (confidence: {confidence:.2f})")

    def on_timeout():
        print("Command timeout")

    try:
        if not detector.initialize():
            print("Failed to initialize wake word detector")
            return False

        detector.set_wake_word_callback(on_wake_word)
        detector.set_command_callback(on_command)
        detector.set_timeout_callback(on_timeout)

        if not detector.start_listening():
            print("Failed to start listening")
            return False

        print(f"Listening for wake words for {duration} seconds...")
        print(f"Wake words: {', '.join(detector.wake_words)}")
        print("Say a wake word followed by a command...")

        time.sleep(duration)

        detector.stop_listening()
        print("Test completed")
        return True

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return True
    except Exception as e:
        print(f"Test error: {e}")
        return False
    finally:
        detector.cleanup()


def check_wake_word_dependencies() -> Dict[str, bool]:
    """Check availability of wake word detection dependencies."""
    return {
        "pyaudio": PYAUDIO_AVAILABLE,
        "vosk": VOSK_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
    }


# Initialize logging
logger.info(
    f"Wake word detector initialized - Dependencies: {check_wake_word_dependencies()}"
)
