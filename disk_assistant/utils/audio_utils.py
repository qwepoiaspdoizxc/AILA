import os
import sys
import wave
import json
import time
import threading
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import logging

# Audio processing imports (with fallbacks)
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.playback import play

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from config import CONFIG
from utils.logger import get_logger

logger = get_logger("AudioUtils")


class AudioRecorder:
    """Audio recording utility with various formats and quality settings."""

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.is_recording = False
        self.audio = None
        self.stream = None
        self.frames = []
        self.recording_thread = None

    def initialize_audio(self) -> bool:
        """Initialize PyAudio for recording."""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available for audio recording")
            return False

        try:
            self.audio = pyaudio.PyAudio()
            logger.info("Audio system initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            return False

    def list_audio_devices(self) -> List[Dict]:
        """List available audio input devices."""
        if not self.audio:
            if not self.initialize_audio():
                return []

        devices = []
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

    def start_recording(
        self,
        sample_rate: int = None,
        channels: int = None,
        format_type: str = None,
        device_index: int = None,
    ) -> bool:
        """Start audio recording."""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False

        if not self.audio:
            if not self.initialize_audio():
                return False

        # Use config or provided parameters
        sample_rate = sample_rate or self.config.get("sample_rate", 44100)
        channels = channels or self.config.get("channels", 1)

        # Convert format type to PyAudio format
        if format_type == "16bit" or format_type is None:
            audio_format = pyaudio.paInt16
            self.sample_width = 2
        elif format_type == "24bit":
            audio_format = pyaudio.paInt24
            self.sample_width = 3
        elif format_type == "32bit":
            audio_format = pyaudio.paInt32
            self.sample_width = 4
        else:
            audio_format = pyaudio.paInt16
            self.sample_width = 2

        try:
            self.stream = self.audio.open(
                format=audio_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
            )

            self.frames = []
            self.is_recording = True
            self.sample_rate = sample_rate
            self.channels = channels

            # Start recording in separate thread
            self.recording_thread = threading.Thread(target=self._record_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()

            logger.info(
                f"Recording started: {sample_rate}Hz, {channels}ch, {format_type}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    def _record_loop(self):
        """Recording loop running in separate thread."""
        while self.is_recording:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                logger.error(f"Recording error: {e}")
                break

    def stop_recording(self) -> bool:
        """Stop audio recording."""
        if not self.is_recording:
            return True

        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        logger.info("Recording stopped")
        return True

    def save_recording(self, output_path: str, audio_format: str = "wav") -> bool:
        """Save recorded audio to file."""
        if not self.frames:
            logger.error("No audio data to save")
            return False

        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if audio_format.lower() == "wav":
                return self._save_as_wav(output_path)
            elif audio_format.lower() in ["mp3", "m4a", "ogg"] and PYDUB_AVAILABLE:
                return self._save_with_pydub(output_path, audio_format)
            else:
                logger.error(f"Unsupported audio format: {audio_format}")
                return False

        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return False

    def _save_as_wav(self, output_path: str) -> bool:
        """Save audio as WAV file."""
        try:
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(self.frames))

            logger.info(f"Audio saved as WAV: {output_path}")
            return True

        except Exception as e:
            logger.error(f"WAV save error: {e}")
            return False

    def _save_with_pydub(self, output_path: str, audio_format: str) -> bool:
        """Save audio using pydub for format conversion."""
        try:
            # First save as temporary WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            if not self._save_as_wav(tmp_path):
                return False

            # Convert using pydub
            audio = AudioSegment.from_wav(tmp_path)
            audio.export(output_path, format=audio_format.lower())

            # Clean up temporary file
            os.unlink(tmp_path)

            logger.info(f"Audio saved as {audio_format.upper()}: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Pydub save error: {e}")
            return False

    def get_recording_duration(self) -> float:
        """Get current recording duration in seconds."""
        if not self.frames or not hasattr(self, "sample_rate"):
            return 0.0

        total_frames = len(self.frames) * 1024  # frames_per_buffer
        return total_frames / self.sample_rate

    def cleanup(self):
        """Clean up audio resources."""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()
            self.audio = None


class AudioPlayer:
    """Audio playback utility with support for multiple formats."""

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.is_playing = False
        self.current_audio = None

    def play_file(self, file_path: str, async_play: bool = True) -> bool:
        """Play audio file."""
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False

        try:
            if PYDUB_AVAILABLE:
                return self._play_with_pydub(file_path, async_play)
            else:
                return self._play_with_system(file_path)

        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False

    def _play_with_pydub(self, file_path: str, async_play: bool) -> bool:
        """Play audio using pydub."""
        try:
            # Load audio file
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".wav":
                audio = AudioSegment.from_wav(file_path)
            elif file_ext == ".mp3":
                audio = AudioSegment.from_mp3(file_path)
            elif file_ext in [".m4a", ".aac"]:
                audio = AudioSegment.from_file(file_path, "m4a")
            else:
                audio = AudioSegment.from_file(file_path)

            self.current_audio = audio

            if async_play:
                threading.Thread(
                    target=self._play_thread, args=(audio,), daemon=True
                ).start()
            else:
                play(audio)

            logger.info(f"Playing audio: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Pydub playback error: {e}")
            return False

    def _play_thread(self, audio):
        """Audio playback thread."""
        try:
            self.is_playing = True
            play(audio)
            self.is_playing = False
        except Exception as e:
            logger.error(f"Audio playback thread error: {e}")
            self.is_playing = False

    def _play_with_system(self, file_path: str) -> bool:
        """Play audio using system commands."""
        try:
            system = sys.platform

            if system.startswith("win"):
                os.system(f'start "" "{file_path}"')
            elif system.startswith("darwin"):
                os.system(f'afplay "{file_path}"')
            elif system.startswith("linux"):
                # Try different players
                for player in ["aplay", "paplay", "play", "vlc"]:
                    if os.system(f"which {player} > /dev/null 2>&1") == 0:
                        os.system(f'{player} "{file_path}" > /dev/null 2>&1 &')
                        break
                else:
                    logger.error("No audio player found on Linux")
                    return False

            logger.info(f"Playing audio with system player: {file_path}")
            return True

        except Exception as e:
            logger.error(f"System audio playback error: {e}")
            return False

    def stop_playback(self):
        """Stop current audio playback."""
        self.is_playing = False
        # Note: Limited stop functionality with pydub
        logger.info("Audio playback stopped")


class AudioAnalyzer:
    """Audio analysis utilities."""

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG

    def get_audio_info(self, file_path: str) -> Dict:
        """Get basic audio file information."""
        if not os.path.exists(file_path):
            return {}

        info = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "format": Path(file_path).suffix.lower()[1:],
        }

        try:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_file(file_path)
                info.update(
                    {
                        "duration": len(audio) / 1000.0,  # seconds
                        "channels": audio.channels,
                        "sample_rate": audio.frame_rate,
                        "sample_width": audio.sample_width,
                        "bitrate": audio.frame_rate
                        * audio.sample_width
                        * 8
                        * audio.channels,
                    }
                )
            elif file_path.endswith(".wav"):
                with wave.open(file_path, "rb") as wf:
                    info.update(
                        {
                            "duration": wf.getnframes() / wf.getframerate(),
                            "channels": wf.getnchannels(),
                            "sample_rate": wf.getframerate(),
                            "sample_width": wf.getsampwidth(),
                        }
                    )

        except Exception as e:
            logger.error(f"Error getting audio info: {e}")

        return info

    def detect_silence(
        self, file_path: str, silence_threshold: float = -40.0
    ) -> List[Tuple[float, float]]:
        """Detect silent segments in audio file."""
        if not PYDUB_AVAILABLE:
            logger.error("Pydub required for silence detection")
            return []

        try:
            audio = AudioSegment.from_file(file_path)

            # Detect silence
            silent_ranges = []
            chunk_length = 100  # ms

            for i in range(0, len(audio), chunk_length):
                chunk = audio[i : i + chunk_length]
                if chunk.dBFS < silence_threshold:
                    start_time = i / 1000.0
                    end_time = min(i + chunk_length, len(audio)) / 1000.0
                    silent_ranges.append((start_time, end_time))

            # Merge consecutive silent ranges
            merged_ranges = []
            for start, end in silent_ranges:
                if merged_ranges and start <= merged_ranges[-1][1]:
                    merged_ranges[-1] = (
                        merged_ranges[-1][0],
                        max(merged_ranges[-1][1], end),
                    )
                else:
                    merged_ranges.append((start, end))

            return merged_ranges

        except Exception as e:
            logger.error(f"Silence detection error: {e}")
            return []

    def extract_features(self, file_path: str) -> Dict:
        """Extract audio features using librosa (if available)."""
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available for advanced audio analysis")
            return self.get_audio_info(file_path)

        try:
            # Load audio
            y, sr = librosa.load(file_path)

            # Extract features
            features = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "rms_energy": float(np.mean(librosa.feature.rms(y=y))),
                "zero_crossing_rate": float(
                    np.mean(librosa.feature.zero_crossing_rate(y))
                ),
                "spectral_centroid": float(
                    np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                ),
                "spectral_rolloff": float(
                    np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                ),
                "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
            }

            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = [float(np.mean(mfcc)) for mfcc in mfccs]

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}


class AudioConverter:
    """Audio format conversion utilities."""

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG

    def convert_format(
        self, input_path: str, output_path: str, target_format: str = "wav", **kwargs
    ) -> bool:
        """Convert audio file to different format."""
        if not PYDUB_AVAILABLE:
            logger.error("Pydub required for audio conversion")
            return False

        try:
            # Load input audio
            audio = AudioSegment.from_file(input_path)

            # Apply conversion parameters
            if "sample_rate" in kwargs:
                audio = audio.set_frame_rate(kwargs["sample_rate"])

            if "channels" in kwargs:
                if kwargs["channels"] == 1:
                    audio = audio.set_channels(1)
                elif kwargs["channels"] == 2:
                    audio = audio.set_channels(2)

            if "bitrate" in kwargs and target_format in ["mp3", "m4a"]:
                export_kwargs = {"bitrate": f"{kwargs['bitrate']}k"}
            else:
                export_kwargs = {}

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Export in target format
            audio.export(output_path, format=target_format, **export_kwargs)

            logger.info(f"Converted {input_path} to {output_path} ({target_format})")
            return True

        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return False

    def normalize_audio(self, input_path: str, output_path: str = None) -> bool:
        """Normalize audio levels."""
        if not PYDUB_AVAILABLE:
            logger.error("Pydub required for audio normalization")
            return False

        output_path = output_path or input_path

        try:
            audio = AudioSegment.from_file(input_path)

            # Normalize to -3dB
            normalized_audio = audio.normalize(headroom=3.0)

            # Save normalized audio
            format_type = Path(output_path).suffix.lower()[1:]
            normalized_audio.export(output_path, format=format_type)

            logger.info(f"Audio normalized: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return False


# Utility functions
def get_supported_formats() -> Dict[str, List[str]]:
    """Get supported audio formats by capability."""
    formats = {"recording": ["wav"], "playback": ["wav"], "conversion": []}

    if PYDUB_AVAILABLE:
        formats["recording"].extend(["mp3", "m4a", "ogg", "flac"])
        formats["playback"].extend(["mp3", "m4a", "ogg", "flac", "aac"])
        formats["conversion"] = ["wav", "mp3", "m4a", "ogg", "flac", "aac"]

    return formats


def create_audio_recorder(config: Dict = None) -> AudioRecorder:
    """Create and return an AudioRecorder instance."""
    return AudioRecorder(config)


def create_audio_player(config: Dict = None) -> AudioPlayer:
    """Create and return an AudioPlayer instance."""
    return AudioPlayer(config)


def create_audio_analyzer(config: Dict = None) -> AudioAnalyzer:
    """Create and return an AudioAnalyzer instance."""
    return AudioAnalyzer(config)


def create_audio_converter(config: Dict = None) -> AudioConverter:
    """Create and return an AudioConverter instance."""
    return AudioConverter(config)


def quick_record_audio(
    output_path: str, duration: float, config: Dict = None, **kwargs
) -> bool:
    """Quick function to record audio for specified duration."""
    recorder = AudioRecorder(config)

    try:
        if not recorder.initialize_audio():
            return False

        if not recorder.start_recording(**kwargs):
            return False

        # Record for specified duration
        time.sleep(duration)

        recorder.stop_recording()

        # Save recording
        audio_format = kwargs.get("format", "wav")
        return recorder.save_recording(output_path, audio_format)

    except Exception as e:
        logger.error(f"Quick record error: {e}")
        return False
    finally:
        recorder.cleanup()


def quick_play_audio(file_path: str, async_play: bool = True) -> bool:
    """Quick function to play audio file."""
    player = AudioPlayer()
    return player.play_file(file_path, async_play)


def get_audio_file_info(file_path: str) -> Dict:
    """Quick function to get audio file information."""
    analyzer = AudioAnalyzer()
    return analyzer.get_audio_info(file_path)


def convert_audio_file(
    input_path: str, output_path: str, target_format: str = "wav", **kwargs
) -> bool:
    """Quick function to convert audio file format."""
    converter = AudioConverter()
    return converter.convert_format(input_path, output_path, target_format, **kwargs)


# Check available dependencies
def check_audio_dependencies() -> Dict[str, bool]:
    """Check which audio processing libraries are available."""
    return {
        "pyaudio": PYAUDIO_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
        "pydub": PYDUB_AVAILABLE,
        "librosa": LIBROSA_AVAILABLE,
    }


# Initialize logging
logger.info(f"Audio utils initialized - Dependencies: {check_audio_dependencies()}")
