"""
AI Components Package

This package contains AI-related modules for the Disk Assistant:
- ai_assistant: Main AI assistant for natural language processing
- model_manager: Manages AI models (local LLMs, embeddings)
- prompt_templates: Templates for AI prompts
- response_formatter: Formats AI responses for different contexts
"""

from .ai_assistant import AIAssistant
from .model_manager import ModelManager, ModelConfig
from .prompt_templates import PromptTemplates
from .response_formatter import ResponseFormatter

__all__ = [
    'AIAssistant',
    'ModelManager', 
    'ModelConfig',
    'PromptTemplates',
    'ResponseFormatter'
]