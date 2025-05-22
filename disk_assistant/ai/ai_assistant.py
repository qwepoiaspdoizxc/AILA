import os
import json
import threading
import queue
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import CONFIG
from utils.logger import logger

# Try importing local LLM libraries
try:
    # For GGUF models (llama.cpp Python bindings)
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning(
        "llama-cpp-python not available. Install with: pip install llama-cpp-python"
    )

try:
    # For Transformers models
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers not available. Install with: pip install transformers torch"
    )


class AIAssistant:
    """AI Assistant for processing natural language queries and generating responses."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_type = CONFIG.get("ai_model_type", "llama")
        self.model_path = CONFIG.get("ai_model_path", "models/local_llm")
        self.max_context_length = CONFIG.get("max_context_length", 4096)
        self.temperature = CONFIG.get("temperature", 0.7)
        self.max_tokens = CONFIG.get("max_tokens", 512)
        self.conversation_history = []
        self.processing_queue = queue.Queue()
        self.is_processing = False

    def initialize(self) -> bool:
        """Initialize the AI model."""
        try:
            if self.model_type.lower() == "llama" and LLAMA_CPP_AVAILABLE:
                return self._initialize_llama_cpp()
            elif (
                self.model_type.lower() in ["gpt", "transformers"]
                and TRANSFORMERS_AVAILABLE
            ):
                return self._initialize_transformers()
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return False
        except Exception as e:
            logger.error(f"Error initializing AI model: {e}")
            return False

    def _initialize_llama_cpp(self) -> bool:
        """Initialize llama.cpp model."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False

            logger.info(f"Loading Llama model from: {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.max_context_length,
                n_threads=CONFIG.get("ai_processing_threads", 2),
                verbose=False,
            )
            logger.info("Llama model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Llama model: {e}")
            return False

    def _initialize_transformers(self) -> bool:
        """Initialize transformers model."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model directory not found: {self.model_path}")
                return False

            logger.info(f"Loading Transformers model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            logger.info("Transformers model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Transformers model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if AI assistant is available and initialized."""
        return self.model is not None

    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate AI response to a prompt."""
        if not self.is_available():
            return "AI assistant is not available. Please check model configuration."

        try:
            # Prepare the full prompt with context
            full_prompt = self._prepare_prompt(prompt, context)

            # Generate response based on model type
            if self.model_type.lower() == "llama" and isinstance(self.model, Llama):
                response = self._generate_llama_response(full_prompt)
            elif self.pipeline:
                response = self._generate_transformers_response(full_prompt)
            else:
                return "Error: No valid model available"

            # Add to conversation history
            self.conversation_history.append(
                {"prompt": prompt, "response": response, "timestamp": time.time()}
            )

            # Keep history limited
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return response
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return f"Error generating response: {str(e)}"

    def _prepare_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Prepare the full prompt with system context and conversation history."""
        system_prompt = self._get_system_prompt()

        # Add context if provided
        context_str = ""
        if context:
            if "search_results" in context:
                context_str += f"\nSearch Results: {json.dumps(context['search_results'], indent=2)}"
            if "file_info" in context:
                context_str += (
                    f"\nFile Information: {json.dumps(context['file_info'], indent=2)}"
                )
            if "user_intent" in context:
                context_str += f"\nUser Intent: {context['user_intent']}"

        # Add recent conversation history
        history_str = ""
        if self.conversation_history:
            history_str = "\nRecent Conversation:\n"
            for item in self.conversation_history[-3:]:  # Last 3 exchanges
                history_str += (
                    f"User: {item['prompt']}\nAssistant: {item['response']}\n"
                )

        full_prompt = (
            f"{system_prompt}{context_str}{history_str}\nUser: {prompt}\nAssistant:"
        )
        return full_prompt

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI assistant."""
        return """You are a helpful AI assistant for a local disk file management system. You help users:
1. Find and organize files on their computer
2. Analyze file contents and metadata
3. Provide insights about their file system
4. Answer questions about files and directories
5. Suggest file management strategies

You have access to file search results and metadata. Be concise, helpful, and accurate.
Focus on practical file management advice and direct answers to user queries."""

    def _generate_llama_response(self, prompt: str) -> str:
        """Generate response using llama.cpp."""
        try:
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["User:", "Human:", "\n\n"],
                echo=False,
            )

            generated_text = response["choices"][0]["text"].strip()
            return generated_text
        except Exception as e:
            logger.error(f"Error with Llama generation: {e}")
            return "Error generating response with Llama model."

    def _generate_transformers_response(self, prompt: str) -> str:
        """Generate response using transformers."""
        try:
            response = self.pipeline(
                prompt,
                max_length=len(prompt.split()) + self.max_tokens,
                temperature=self.temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated_text = response[0]["generated_text"]
            # Extract only the new generated part
            new_text = generated_text[len(prompt) :].strip()

            # Clean up the response
            if "User:" in new_text:
                new_text = new_text.split("User:")[0].strip()

            return new_text
        except Exception as e:
            logger.error(f"Error with Transformers generation: {e}")
            return "Error generating response with Transformers model."

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and required actions."""
        intent_prompt = f"""Analyze this user query and determine the intent and required actions:
Query: "{query}"

Respond with a JSON object containing:
- intent: (search, analyze, organize, help, open, delete, etc.)
- action_type: (file_search, data_analysis, file_operation, etc.)
- parameters: (relevant parameters for the action)
- confidence: (0.0 to 1.0)

Response:"""

        try:
            response = self.generate_response(intent_prompt)
            # Try to parse JSON response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback analysis
                return self._fallback_intent_analysis(query)
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return self._fallback_intent_analysis(query)

    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback intent analysis without AI."""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["find", "search", "look for", "where is"]
        ):
            return {
                "intent": "search",
                "action_type": "file_search",
                "parameters": {"query": query},
                "confidence": 0.8,
            }
        elif any(
            word in query_lower for word in ["analyze", "statistics", "chart", "graph"]
        ):
            return {
                "intent": "analyze",
                "action_type": "data_analysis",
                "parameters": {"query": query},
                "confidence": 0.7,
            }
        elif any(word in query_lower for word in ["open", "launch", "start"]):
            return {
                "intent": "open",
                "action_type": "file_operation",
                "parameters": {"query": query},
                "confidence": 0.8,
            }
        else:
            return {
                "intent": "help",
                "action_type": "general_query",
                "parameters": {"query": query},
                "confidence": 0.5,
            }

    def process_file_context(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process file context for AI analysis."""
        context = {
            "file_count": len(file_paths),
            "file_types": {},
            "total_size": 0,
            "recent_files": [],
        }

        for file_path in file_paths[:10]:  # Limit to first 10 files
            try:
                path_obj = Path(file_path)
                if path_obj.exists():
                    stat = path_obj.stat()
                    file_type = path_obj.suffix.lower()

                    context["file_types"][file_type] = (
                        context["file_types"].get(file_type, 0) + 1
                    )
                    context["total_size"] += stat.st_size

                    context["recent_files"].append(
                        {
                            "name": path_obj.name,
                            "path": str(path_obj),
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "type": file_type,
                        }
                    )
            except Exception as e:
                logger.debug(f"Error processing file context for {file_path}: {e}")

        return context

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("AI conversation history cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "is_available": self.is_available(),
            "max_context_length": self.max_context_length,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "conversation_count": len(self.conversation_history),
        }


class AIResponseFormatter:
    """Format AI responses for different output modes."""

    @staticmethod
    def format_search_response(
        results: List[Dict], query: str, ai_response: str
    ) -> str:
        """Format search results with AI commentary."""
        if not results:
            return f"No files found for '{query}'. {ai_response}"

        formatted = f"Found {len(results)} files for '{query}':\n\n"

        for i, result in enumerate(results[:5], 1):  # Show top 5
            formatted += f"{i}. {result.get('filename', 'Unknown')}\n"
            formatted += f"   Path: {result.get('path', 'Unknown')}\n"
            formatted += f"   Type: {result.get('file_type', 'Unknown')}\n\n"

        if len(results) > 5:
            formatted += f"... and {len(results) - 5} more files.\n\n"

        formatted += f"AI Analysis: {ai_response}"
        return formatted

    @staticmethod
    def format_analysis_response(analysis_result: Dict, ai_response: str) -> str:
        """Format data analysis results with AI commentary."""
        formatted = "Data Analysis Results:\n\n"

        if "summary" in analysis_result:
            formatted += f"Summary: {analysis_result['summary']}\n\n"

        if "charts" in analysis_result:
            formatted += f"Generated Charts: {len(analysis_result['charts'])}\n"
            for chart in analysis_result["charts"]:
                formatted += f"  - {chart.get('title', 'Chart')}: {chart.get('path', 'Unknown')}\n"
            formatted += "\n"

        formatted += f"AI Insights: {ai_response}"
        return formatted
