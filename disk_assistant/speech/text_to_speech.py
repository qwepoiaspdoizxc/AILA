import os
import sys
import asyncio
import threading
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from queue import Queue, Empty
import time

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger("TextToSpeech")


class TTSEngine:
    """Base class for TTS engines."""

    def __init__(self, config: Dict):
        self.config = config
        self.is_speaking = False

    def speak(self, text: str, **kwargs) -> bool:
        """Speak the given text. Returns True if successful."""
        raise NotImplementedError

    def speak_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        """Save speech to file. Returns True if successful."""
        raise NotImplementedError

    def stop(self):
        """Stop current speech."""
        raise NotImplementedError

    def set_voice_properties(self, **kwargs):
        """Set voice properties like rate, volume, voice."""
        raise NotImplementedError


class SystemTTSEngine(TTSEngine):
    """System-based TTS engine using OS native TTS."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.system = sys.platform

    def speak(self, text: str, **kwargs) -> bool:
        """Use system TTS to speak text."""
        try:
            self.is_speaking = True

            if self.system.startswith("win"):
                # Windows SAPI
                os.system(
                    f"powershell -Command \"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
                )
            elif self.system.startswith("darwin"):
                # macOS
                os.system(f"say '{text}'")
            elif self.system.startswith("linux"):
                # Linux - try espeak or festival
                if os.system("which espeak > /dev/null 2>&1") == 0:
                    os.system(f"espeak '{text}'")
                elif os.system("which festival > /dev/null 2>&1") == 0:
                    os.system(f"echo '{text}' | festival --tts")
                else:
                    logger.warning("No TTS engine found on Linux")
                    return False
            else:
                logger.warning(f"Unsupported system: {self.system}")
                return False

            self.is_speaking = False
            return True

        except Exception as e:
            logger.error(f"System TTS error: {e}")
            self.is_speaking = False
            return False

    def speak_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        """Save speech to file using system TTS."""
        try:
            if self.system.startswith("win"):
                # Windows SAPI to WAV
                script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("{output_path}")
$synth.Speak("{text}")
$synth.Dispose()
'''
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".ps1", delete=False
                ) as f:
                    f.write(script)
                    script_path = f.name

                result = os.system(
                    f'powershell -ExecutionPolicy Bypass -File "{script_path}"'
                )
                os.unlink(script_path)
                return result == 0

            elif self.system.startswith("darwin"):
                # macOS to AIFF
                result = os.system(f"say '{text}' -o '{output_path}'")
                return result == 0

            elif self.system.startswith("linux"):
                # Linux espeak to WAV
                if os.system("which espeak > /dev/null 2>&1") == 0:
                    result = os.system(f"espeak '{text}' -w '{output_path}'")
                    return result == 0

            return False

        except Exception as e:
            logger.error(f"System TTS file output error: {e}")
            return False

    def stop(self):
        """Stop system TTS (limited support)."""
        self.is_speaking = False
        if self.system.startswith("win"):
            os.system('taskkill /f /im "WindowsMediaPlayer.exe" > nul 2>&1')


class Pyttsx3Engine(TTSEngine):
    """Pyttsx3-based TTS engine."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.engine = None
        self.init_engine()

    def init_engine(self):
        """Initialize pyttsx3 engine."""
        try:
            if PYTTSX3_AVAILABLE:
                self.engine = pyttsx3.init()
                self._configure_voice()
                logger.info("Pyttsx3 TTS engine initialized")
            else:
                logger.warning("Pyttsx3 not available")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            self.engine = None

    def _configure_voice(self):
        """Configure voice properties."""
        if not self.engine:
            return

        try:
            # Set speech rate
            rate = self.config.get("voice_speed", 150)
            self.engine.setProperty("rate", rate)

            # Set volume
            volume = self.config.get("voice_volume", 0.8)
            self.engine.setProperty("volume", volume)

            # Set voice if specified
            voice_id = self.config.get("voice_id")
            if voice_id:
                voices = self.engine.getProperty("voices")
                if voices and len(voices) > voice_id:
                    self.engine.setProperty("voice", voices[voice_id].id)

        except Exception as e:
            logger.error(f"Voice configuration error: {e}")

    def speak(self, text: str, **kwargs) -> bool:
        """Speak text using pyttsx3."""
        if not self.engine:
            return False

        try:
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
            return True

        except Exception as e:
            logger.error(f"Pyttsx3 speak error: {e}")
            self.is_speaking = False
            return False

    def speak_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        """Save speech to file using pyttsx3."""
        if not self.engine:
            return False

        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            return os.path.exists(output_path)

        except Exception as e:
            logger.error(f"Pyttsx3 file output error: {e}")
            return False

    def stop(self):
        """Stop pyttsx3 speech."""
        if self.engine:
            try:
                self.engine.stop()
                self.is_speaking = False
            except Exception as e:
                logger.error(f"Pyttsx3 stop error: {e}")

    def set_voice_properties(self, **kwargs):
        """Set voice properties."""
        if not self.engine:
            return

        try:
            if "rate" in kwargs:
                self.engine.setProperty("rate", kwargs["rate"])
            if "volume" in kwargs:
                self.engine.setProperty("volume", kwargs["volume"])
            if "voice_id" in kwargs:
                voices = self.engine.getProperty("voices")
                if voices and len(voices) > kwargs["voice_id"]:
                    self.engine.setProperty("voice", voices[kwargs["voice_id"]].id)

        except Exception as e:
            logger.error(f"Set voice properties error: {e}")


class EdgeTTSEngine(TTSEngine):
    """Microsoft Edge TTS engine (cloud-based)."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.voice = config.get("edge_voice", "en-US-JennyNeural")

    async def _speak_async(self, text: str, output_path: Optional[str] = None) -> bool:
        """Async speech generation."""
        try:
            if not EDGE_TTS_AVAILABLE:
                return False

            communicate = edge_tts.Communicate(text, self.voice)

            if output_path:
                await communicate.save(output_path)
                return os.path.exists(output_path)
            else:
                # Stream to temporary file and play
                with tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                ) as tmp_file:
                    await communicate.save(tmp_file.name)

                    # Play the file
                    if PYGAME_AVAILABLE:
                        pygame.mixer.init()
                        pygame.mixer.music.load(tmp_file.name)
                        pygame.mixer.music.play()

                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.1)

                        pygame.mixer.quit()

                    # Clean up
                    os.unlink(tmp_file.name)
                    return True

        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return False

    def speak(self, text: str, **kwargs) -> bool:
        """Speak text using Edge TTS."""
        try:
            self.is_speaking = True
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._speak_async(text))
            loop.close()
            self.is_speaking = False
            return result

        except Exception as e:
            logger.error(f"Edge TTS speak error: {e}")
            self.is_speaking = False
            return False

    def speak_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        """Save speech to file using Edge TTS."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._speak_async(text, output_path))
            loop.close()
            return result

        except Exception as e:
            logger.error(f"Edge TTS file output error: {e}")
            return False

    def stop(self):
        """Stop Edge TTS (limited support)."""
        self.is_speaking = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except:
                pass


class TextToSpeech:
    """Main TTS interface with support for multiple engines."""

    def __init__(self, config: Dict):
        self.config = config
        self.current_engine = None
        self.available_engines = {}
        self.speech_queue = Queue()
        self.queue_thread = None
        self.queue_running = False

        # Initialize engines
        self._init_engines()

        # Set default engine
        preferred_engine = config.get("voice_engine", "system")
        self.set_engine(preferred_engine)

        # Start queue processing
        self._start_queue_processing()

        logger.info(f"TTS initialized with {len(self.available_engines)} engines")

    def _init_engines(self):
        """Initialize available TTS engines."""
        # System TTS
        try:
            self.available_engines["system"] = SystemTTSEngine(self.config)
            logger.info("System TTS engine available")
        except Exception as e:
            logger.error(f"System TTS initialization failed: {e}")

        # Pyttsx3
        if PYTTSX3_AVAILABLE:
            try:
                self.available_engines["pyttsx3"] = Pyttsx3Engine(self.config)
                logger.info("Pyttsx3 TTS engine available")
            except Exception as e:
                logger.error(f"Pyttsx3 initialization failed: {e}")

        # Edge TTS
        if EDGE_TTS_AVAILABLE:
            try:
                self.available_engines["edge"] = EdgeTTSEngine(self.config)
                logger.info("Edge TTS engine available")
            except Exception as e:
                logger.error(f"Edge TTS initialization failed: {e}")

    def set_engine(self, engine_name: str) -> bool:
        """Set the active TTS engine."""
        if engine_name in self.available_engines:
            self.current_engine = self.available_engines[engine_name]
            logger.info(f"TTS engine set to: {engine_name}")
            return True
        else:
            logger.warning(f"TTS engine '{engine_name}' not available")
            return False

    def get_available_engines(self) -> List[str]:
        """Get list of available engine names."""
        return list(self.available_engines.keys())

    def speak(self, text: str, priority: bool = False, **kwargs) -> bool:
        """Speak text immediately or add to queue."""
        if not self.current_engine:
            logger.error("No TTS engine available")
            return False

        if not text or not text.strip():
            return True

        # Clean text
        clean_text = self._clean_text(text)

        if priority:
            # Speak immediately
            return self._speak_immediately(clean_text, **kwargs)
        else:
            # Add to queue
            self.speech_queue.put(
                {"text": clean_text, "kwargs": kwargs, "timestamp": time.time()}
            )
            return True

    def speak_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        """Save speech to audio file."""
        if not self.current_engine:
            logger.error("No TTS engine available")
            return False

        clean_text = self._clean_text(text)

        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            result = self.current_engine.speak_to_file(
                clean_text, output_path, **kwargs
            )

            if result:
                logger.info(f"Speech saved to: {output_path}")
            else:
                logger.error(f"Failed to save speech to: {output_path}")

            return result

        except Exception as e:
            logger.error(f"Speech file save error: {e}")
            return False

    def _speak_immediately(self, text: str, **kwargs) -> bool:
        """Speak text immediately, bypassing queue."""
        try:
            return self.current_engine.speak(text, **kwargs)
        except Exception as e:
            logger.error(f"Immediate speech error: {e}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean text for better TTS output."""
        # Remove markdown formatting
        import re

        # Remove markdown
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # Italic
        text = re.sub(r"`(.*?)`", r"\1", text)  # Code
        text = re.sub(r"#{1,6}\s*(.*?)$", r"\1", text, flags=re.MULTILINE)  # Headers

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "link",
            text,
        )

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Handle special characters that might cause issues
        text = text.replace('"', "'")
        text = text.replace("\n", ". ")
        text = text.replace("\t", " ")

        return text

    def _start_queue_processing(self):
        """Start the speech queue processing thread."""
        if not self.queue_running:
            self.queue_running = True
            self.queue_thread = threading.Thread(
                target=self._process_queue, daemon=True
            )
            self.queue_thread.start()
            logger.info("Speech queue processing started")

    def _process_queue(self):
        """Process speech queue in background thread."""
        while self.queue_running:
            try:
                # Get speech request with timeout
                speech_request = self.speech_queue.get(timeout=1.0)

                if self.current_engine and not self.current_engine.is_speaking:
                    text = speech_request["text"]
                    kwargs = speech_request.get("kwargs", {})

                    logger.debug(f"Processing queued speech: {text[:50]}...")
                    self._speak_immediately(text, **kwargs)

                self.speech_queue.task_done()

            except Empty:
                # Timeout, continue loop
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def stop_speech(self):
        """Stop current speech and clear queue."""
        if self.current_engine:
            self.current_engine.stop()

        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except Empty:
                break

        logger.info("Speech stopped and queue cleared")

    def is_speaking(self) -> bool:
        """Check if TTS is currently speaking."""
        return self.current_engine.is_speaking if self.current_engine else False

    def queue_size(self) -> int:
        """Get current queue size."""
        return self.speech_queue.qsize()

    def set_voice_properties(self, **kwargs):
        """Set voice properties for current engine."""
        if self.current_engine:
            self.current_engine.set_voice_properties(**kwargs)

    def get_voices(self) -> List[Dict]:
        """Get available voices (engine-dependent)."""
        if (
            isinstance(self.current_engine, Pyttsx3Engine)
            and self.current_engine.engine
        ):
            try:
                voices = self.current_engine.engine.getProperty("voices")
                return [
                    {
                        "id": i,
                        "name": voice.name,
                        "languages": getattr(voice, "languages", []),
                    }
                    for i, voice in enumerate(voices)
                ]
            except Exception as e:
                logger.error(f"Get voices error: {e}")

        return []

    def shutdown(self):
        """Shutdown TTS system."""
        logger.info("Shutting down TTS system")

        # Stop queue processing
        self.queue_running = False
        if self.queue_thread:
            self.queue_thread.join(timeout=2.0)

        # Stop current speech
        self.stop_speech()

        # Clean up engines
        for engine in self.available_engines.values():
            try:
                engine.stop()
            except:
                pass

        logger.info("TTS system shutdown complete")


# Convenience functions for easy usage
def create_tts(config: Dict) -> TextToSpeech:
    """Create and return a TTS instance."""
    return TextToSpeech(config)


def speak_text(text: str, config: Dict, engine: str = "system") -> bool:
    """Quick function to speak text."""
    try:
        tts = TextToSpeech(config)
        tts.set_engine(engine)
        return tts.speak(text, priority=True)
    except Exception as e:
        logger.error(f"Quick speak error: {e}")
        return False


def text_to_audio_file(
    text: str, output_path: str, config: Dict, engine: str = "system"
) -> bool:
    """Quick function to convert text to audio file."""
    try:
        tts = TextToSpeech(config)
        tts.set_engine(engine)
        return tts.speak_to_file(text, output_path)
    except Exception as e:
        logger.error(f"Quick text-to-file error: {e}")
        return False
