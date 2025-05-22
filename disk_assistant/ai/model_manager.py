"""
AI Model Management

Handles loading, caching, and management of AI models including:
- Local LLM models (GGUF, HuggingFace)
- Embedding models
- Model switching and configuration
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from config import CONFIG
from utils.logger import logger

# Try importing AI libraries
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for an AI model."""

    name: str
    model_type: str  # 'llama', 'transformers', 'sentence_transformer'
    model_path: str
    max_context_length: int = 4096
    temperature: float = 0.7
    max_tokens: int = 512
    device: str = "auto"
    precision: str = "float16"
    is_loaded: bool = False
    load_time: Optional[datetime] = None
    memory_usage: Optional[int] = None


class ModelManager:
    """Manages AI models including loading, caching, and switching."""

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.current_model: Optional[str] = None
        self.models_dir = Path(CONFIG.get("ai_model_path", "models"))
        self.lock = threading.Lock()

        # Initialize model configurations
        self._initialize_model_configs()

    def _initialize_model_configs(self):
        """Initialize available model configurations."""
        # Default LLM model
        default_llm_path = self.models_dir / "local_llm"
        if default_llm_path.exists():
            self.models["default_llm"] = ModelConfig(
                name="default_llm",
                model_type="llama"
                if any(default_llm_path.glob("*.gguf"))
                else "transformers",
                model_path=str(default_llm_path),
                max_context_length=CONFIG.get("max_context_length", 4096),
                temperature=CONFIG.get("temperature", 0.7),
                max_tokens=CONFIG.get("max_tokens", 512),
            )

        # Embedding model
        embedding_path = self.models_dir / "embeddings"
        if embedding_path.exists():
            self.models["embeddings"] = ModelConfig(
                name="embeddings",
                model_type="sentence_transformer",
                model_path=str(embedding_path),
                max_context_length=512,
            )

        # Scan for additional models
        self._scan_for_models()

    def _scan_for_models(self):
        """Scan models directory for additional models."""
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            return

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            if model_name in self.models:
                continue

            # Detect model type based on files
            if any(model_dir.glob("*.gguf")):
                model_type = "llama"
            elif (model_dir / "config.json").exists() and (
                model_dir / "pytorch_model.bin"
            ).exists():
                model_type = "transformers"
            elif (model_dir / "sentence_bert_config.json").exists():
                model_type = "sentence_transformer"
            else:
                continue

            self.models[model_name] = ModelConfig(
                name=model_name, model_type=model_type, model_path=str(model_dir)
            )

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())

    def get_model_info(self, model_name: str) -> Optional[ModelConfig]:
        """Get information about a specific model."""
        return self.models.get(model_name)

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        return model_name in self.loaded_models

    def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """Load a specific model."""
        if model_name not in self.models:
            logger.error(f"Model '{model_name}' not found in available models")
            return False

        with self.lock:
            # Check if already loaded
            if not force_reload and self.is_model_loaded(model_name):
                logger.info(f"Model '{model_name}' is already loaded")
                self.current_model = model_name
                return True

            model_config = self.models[model_name]
            logger.info(
                f"Loading model '{model_name}' of type '{model_config.model_type}'"
            )

            try:
                start_time = datetime.now()

                if model_config.model_type == "llama":
                    success = self._load_llama_model(model_name, model_config)
                elif model_config.model_type == "transformers":
                    success = self._load_transformers_model(model_name, model_config)
                elif model_config.model_type == "sentence_transformer":
                    success = self._load_sentence_transformer_model(
                        model_name, model_config
                    )
                else:
                    logger.error(f"Unsupported model type: {model_config.model_type}")
                    return False

                if success:
                    model_config.is_loaded = True
                    model_config.load_time = datetime.now()
                    self.current_model = model_name

                    load_duration = (datetime.now() - start_time).total_seconds()
                    logger.info(
                        f"Model '{model_name}' loaded successfully in {load_duration:.2f}s"
                    )
                    return True
                else:
                    logger.error(f"Failed to load model '{model_name}'")
                    return False

            except Exception as e:
                logger.error(f"Error loading model '{model_name}': {e}")
                return False

    def _load_llama_model(self, model_name: str, config: ModelConfig) -> bool:
        """Load a Llama.cpp model."""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available")
            return False

        try:
            # Find GGUF file
            model_path = Path(config.model_path)
            gguf_files = list(model_path.glob("*.gguf"))

            if not gguf_files:
                logger.error(f"No GGUF files found in {model_path}")
                return False

            gguf_file = gguf_files[0]  # Use first GGUF file

            model = Llama(
                model_path=str(gguf_file),
                n_ctx=config.max_context_length,
                n_threads=CONFIG.get("ai_processing_threads", 2),
                verbose=False,
            )

            self.loaded_models[model_name] = model
            return True

        except Exception as e:
            logger.error(f"Error loading Llama model: {e}")
            return False

    def _load_transformers_model(self, model_name: str, config: ModelConfig) -> bool:
        """Load a HuggingFace Transformers model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available")
            return False

        try:
            # Determine device and dtype
            device = (
                "cuda"
                if torch.cuda.is_available() and config.device != "cpu"
                else "cpu"
            )
            dtype = (
                torch.float16
                if config.precision == "float16" and device == "cuda"
                else torch.float32
            )

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )

            if device == "cpu":
                model = model.to(device)

            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            return True

        except Exception as e:
            logger.error(f"Error loading Transformers model: {e}")
            return False

    def _load_sentence_transformer_model(
        self, model_name: str, config: ModelConfig
    ) -> bool:
        """Load a Sentence Transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return False

        try:
            model = SentenceTransformer(config.model_path)
            self.loaded_models[model_name] = model
            return True

        except Exception as e:
            logger.error(f"Error loading Sentence Transformer model: {e}")
            return False

    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free memory."""
        with self.lock:
            if model_name not in self.loaded_models:
                logger.warning(f"Model '{model_name}' is not loaded")
                return False

            try:
                # Clean up model
                del self.loaded_models[model_name]

                if model_name in self.tokenizers:
                    del self.tokenizers[model_name]

                # Update config
                if model_name in self.models:
                    self.models[model_name].is_loaded = False

                # Clear current model if it was the unloaded one
                if self.current_model == model_name:
                    self.current_model = None

                # Force garbage collection
                import gc

                gc.collect()

                logger.info(f"Model '{model_name}' unloaded successfully")
                return True

            except Exception as e:
                logger.error(f"Error unloading model '{model_name}': {e}")
                return False

    def get_current_model(self) -> Optional[Any]:
        """Get the currently active model."""
        if not self.current_model or self.current_model not in self.loaded_models:
            return None
        return self.loaded_models[self.current_model]

    def get_current_tokenizer(self) -> Optional[Any]:
        """Get the tokenizer for the current model."""
        if not self.current_model or self.current_model not in self.tokenizers:
            return None
        return self.tokenizers[self.current_model]

    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if model_name == self.current_model:
            return True

        if not self.load_model(model_name):
            return False

        self.current_model = model_name
        logger.info(f"Switched to model: {model_name}")
        return True

    def generate_text(
        self, prompt: str, model_name: Optional[str] = None, **kwargs
    ) -> str:
        """Generate text using the specified or current model."""
        target_model = model_name or self.current_model

        if not target_model or target_model not in self.loaded_models:
            return "No model available for text generation"

        model_config = self.models[target_model]
        model = self.loaded_models[target_model]

        try:
            if model_config.model_type == "llama":
                return self._generate_with_llama(model, prompt, model_config, **kwargs)
            elif model_config.model_type == "transformers":
                return self._generate_with_transformers(
                    model, target_model, prompt, model_config, **kwargs
                )
            else:
                return "Model type does not support text generation"

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"

    def _generate_with_llama(
        self, model, prompt: str, config: ModelConfig, **kwargs
    ) -> str:
        """Generate text using Llama model."""
        response = model(
            prompt,
            max_tokens=kwargs.get("max_tokens", config.max_tokens),
            temperature=kwargs.get("temperature", config.temperature),
            stop=kwargs.get("stop", ["User:", "Human:", "\n\n"]),
            echo=False,
        )
        return response["choices"][0]["text"].strip()

    def _generate_with_transformers(
        self, model, model_name: str, prompt: str, config: ModelConfig, **kwargs
    ) -> str:
        """Generate text using Transformers model."""
        tokenizer = self.tokenizers[model_name]

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1]
                + kwargs.get("max_tokens", config.max_tokens),
                temperature=kwargs.get("temperature", config.temperature),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new generated part
        new_text = generated_text[len(prompt) :].strip()
        return new_text

    def get_embeddings(
        self, texts: Union[str, List[str]], model_name: Optional[str] = None
    ) -> Optional[List[List[float]]]:
        """Get embeddings for text(s) using embedding model."""
        # Find embedding model
        embedding_model_name = model_name
        if not embedding_model_name:
            for name, config in self.models.items():
                if config.model_type == "sentence_transformer":
                    embedding_model_name = name
                    break

        if not embedding_model_name:
            logger.error("No embedding model available")
            return None

        if not self.is_model_loaded(embedding_model_name):
            if not self.load_model(embedding_model_name):
                return None

        try:
            model = self.loaded_models[embedding_model_name]
            if isinstance(texts, str):
                texts = [texts]

            embeddings = model.encode(texts)
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def get_memory_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get memory usage information for loaded models."""
        usage_info = {}

        for model_name in self.loaded_models:
            try:
                # Basic memory info
                model_info = {
                    "loaded": True,
                    "model_type": self.models[model_name].model_type,
                    "load_time": self.models[model_name].load_time.isoformat()
                    if self.models[model_name].load_time
                    else None,
                }

                # Try to get more detailed memory info
                if torch and TRANSFORMERS_AVAILABLE:
                    try:
                        import psutil

                        process = psutil.Process()
                        model_info["memory_mb"] = (
                            process.memory_info().rss / 1024 / 1024
                        )
                    except ImportError:
                        model_info["memory_mb"] = "Unknown"

                usage_info[model_name] = model_info

            except Exception as e:
                logger.debug(f"Error getting memory info for {model_name}: {e}")
                usage_info[model_name] = {"loaded": True, "error": str(e)}

        return usage_info

    def cleanup(self):
        """Clean up all loaded models."""
        with self.lock:
            for model_name in list(self.loaded_models.keys()):
                self.unload_model(model_name)

            self.current_model = None
            logger.info("All models cleaned up")

    def save_model_config(self, config_path: Optional[str] = None):
        """Save model configurations to file."""
        if not config_path:
            config_path = self.models_dir / "model_config.json"

        try:
            config_data = {}
            for name, config in self.models.items():
                config_data[name] = {
                    "name": config.name,
                    "model_type": config.model_type,
                    "model_path": config.model_path,
                    "max_context_length": config.max_context_length,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "device": config.device,
                    "precision": config.precision,
                }

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Model configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Error saving model config: {e}")

    def load_model_config(self, config_path: Optional[str] = None):
        """Load model configurations from file."""
        if not config_path:
            config_path = self.models_dir / "model_config.json"

        if not Path(config_path).exists():
            return

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            for name, config in config_data.items():
                self.models[name] = ModelConfig(**config)

            logger.info(f"Model configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Error loading model config: {e}")
