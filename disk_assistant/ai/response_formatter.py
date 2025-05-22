"""
AI Response Formatter

Formats AI responses for different output modes and contexts including:
- Search results formatting
- Data analysis presentation
- Error handling
- Voice response adaptation
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from utils.logger import logger


class ResponseFormatter:
    """Format AI responses for different contexts and output modes."""

    def __init__(self):
        self.max_results_display = 10
        self.max_summary_length = 200

    def format_search_response(
        self,
        ai_response: str,
        results: List[Dict],
        query: str,
        response_mode: str = "text",
    ) -> str:
        """Format search response with results."""
        if not results:
            formatted = f"🔍 **No files found for '{query}'**\n\n{ai_response}"
            return self._adapt_for_mode(formatted, response_mode)

        # Header
        count = len(results)
        formatted = (
            f"🔍 **Found {count} file{'s' if count != 1 else ''} for '{query}'**\n\n"
        )

        # AI commentary first
        formatted += f"**AI Analysis:** {ai_response}\n\n"

        # Results summary
        if count <= self.max_results_display:
            formatted += "📁 **All Results:**\n"
            for i, result in enumerate(results, 1):
                formatted += self._format_single_result(result, i)
        else:
            formatted += f"📁 **Top {self.max_results_display} Results:**\n"
            for i, result in enumerate(results[: self.max_results_display], 1):
                formatted += self._format_single_result(result, i)
            formatted += (
                f"\n*... and {count - self.max_results_display} more results*\n"
            )

        return self._adapt_for_mode(formatted, response_mode)

    def format_data_analysis_response(
        self, ai_response: str, analysis_result: Dict, response_mode: str = "text"
    ) -> str:
        """Format data analysis results with AI commentary."""
        formatted = "📊 **Data Analysis Complete**\n\n"

        # AI insights first
        formatted += f"**AI Insights:** {ai_response}\n\n"

        # Basic statistics
        if "basic_stats" in analysis_result:
            stats = analysis_result["basic_stats"]
            formatted += "📈 **Dataset Overview:**\n"
            formatted += f"• Rows: {stats.get('shape', [0, 0])[0]:,}\n"
            formatted += f"• Columns: {stats.get('shape', [0, 0])[1]}\n"
            formatted += (
                f"• Memory Usage: {self._format_bytes(stats.get('memory_usage', 0))}\n"
            )

            if stats.get("duplicate_rows", 0) > 0:
                formatted += f"• Duplicate Rows: {stats['duplicate_rows']:,}\n"
            formatted += "\n"

        # Key insights
        if "insights" in analysis_result:
            formatted += "💡 **Key Insights:**\n"
            for insight in analysis_result["insights"][:5]:  # Top 5 insights
                formatted += f"• {insight}\n"
            formatted += "\n"

        # Generated charts
        if "charts" in analysis_result and analysis_result["charts"]:
            formatted += "📈 **Generated Visualizations:**\n"
            for chart in analysis_result["charts"]:
                chart_name = Path(chart.get("path", "")).stem
                formatted += f"• {chart.get('title', chart_name)}\n"
            formatted += "\n"

        # Recommendations
        if "recommendations" in analysis_result:
            formatted += "🎯 **Recommendations:**\n"
            for rec in analysis_result["recommendations"][:3]:  # Top 3 recommendations
                formatted += f"• {rec}\n"

        return self._adapt_for_mode(formatted, response_mode)

    def format_file_operation_response(
        self,
        ai_response: str,
        operation: str,
        files: List[str],
        success: bool,
        response_mode: str = "text",
    ) -> str:
        """Format file operation results."""
        icon = "✅" if success else "❌"
        status = "completed successfully" if success else "failed"

        formatted = f"{icon} **File {operation.title()} {status}**\n\n"
        formatted += f"**AI Response:** {ai_response}\n\n"

        if files:
            formatted += f"**Files Processed ({len(files)}):**\n"
            for file_path in files[:5]:  # Show first 5 files
                filename = Path(file_path).name
                formatted += f"• {filename}\n"

            if len(files) > 5:
                formatted += f"• ... and {len(files) - 5} more files\n"

        return self._adapt_for_mode(formatted, response_mode)

    def format_error_response(
        self,
        ai_response: str,
        error_type: str,
        details: Optional[str] = None,
        response_mode: str = "text",
    ) -> str:
        """Format error response."""
        icon = "⚠️" if error_type == "warning" else "❌"
        formatted = f"{icon} **{error_type.title()} Encountered**\n\n"
        formatted += f"{ai_response}\n"

        if details:
            formatted += f"\n*Technical Details: {details}*"

        return self._adapt_for_mode(formatted, response_mode)

    def format_help_response(
        self,
        ai_response: str,
        help_category: Optional[str] = None,
        response_mode: str = "text",
    ) -> str:
        """Format help response."""
        if help_category:
            formatted = f"💡 **Help - {help_category.title()}**\n\n"
        else:
            formatted = "💡 **Assistant Help**\n\n"

        formatted += ai_response

        return self._adapt_for_mode(formatted, response_mode)

    def format_system_status_response(
        self, ai_response: str, system_info: Dict, response_mode: str = "text"
    ) -> str:
        """Format system status information."""
        formatted = "🖥️ **System Status**\n\n"
        formatted += f"**AI Analysis:** {ai_response}\n\n"

        # Indexing status
        if "indexing" in system_info:
            idx_info = system_info["indexing"]
            formatted += "📁 **File Index:**\n"
            formatted += f"• Total Files: {idx_info.get('total_files', 0):,}\n"
            formatted += f"• Last Updated: {idx_info.get('last_update', 'Unknown')}\n"
            formatted += (
                f"• Index Size: {self._format_bytes(idx_info.get('index_size', 0))}\n\n"
            )

        # AI Model status
        if "ai_model" in system_info:
            model_info = system_info["ai_model"]
            formatted += "🤖 **AI Model:**\n"
            formatted += f"• Status: {model_info.get('status', 'Unknown')}\n"
            formatted += f"• Type: {model_info.get('type', 'Unknown')}\n"
            formatted += f"• Memory Usage: {self._format_bytes(model_info.get('memory_usage', 0))}\n\n"

        # Performance metrics
        if "performance" in system_info:
            perf_info = system_info["performance"]
            formatted += "⚡ **Performance:**\n"
            formatted += f"• CPU Usage: {perf_info.get('cpu_percent', 0):.1f}%\n"
            formatted += f"• Memory Usage: {perf_info.get('memory_percent', 0):.1f}%\n"
            formatted += f"• Disk Usage: {perf_info.get('disk_percent', 0):.1f}%\n"

        return self._adapt_for_mode(formatted, response_mode)

    def format_conversation_response(
        self,
        ai_response: str,
        context: Optional[Dict] = None,
        response_mode: str = "text",
    ) -> str:
        """Format general conversation response."""
        # For general conversation, just return the AI response with minimal formatting
        formatted = f"🤖 {ai_response}"
        return self._adapt_for_mode(formatted, response_mode)

    def _format_single_result(self, result: Dict, index: int) -> str:
        """Format a single search result."""
        filename = result.get("filename", "Unknown")
        path = result.get("path", "")
        file_type = result.get("file_type", "unknown")

        # Try to get a more readable path
        try:
            rel_path = str(Path(path).relative_to(Path.home()))
            if len(rel_path) > 60:
                rel_path = "..." + rel_path[-57:]
        except:
            rel_path = path
            if len(rel_path) > 60:
                rel_path = "..." + rel_path[-57:]

        formatted = f"{index}. **{filename}** ({file_type})\n"
        formatted += f"   📂 {rel_path}\n"

        # Add modification time if available
        if "last_modified" in result:
            try:
                if isinstance(result["last_modified"], str):
                    mod_time = result["last_modified"]
                else:
                    mod_time = result["last_modified"].strftime("%Y-%m-%d %H:%M")
                formatted += f"   🕒 Modified: {mod_time}\n"
            except:
                pass

        formatted += "\n"
        return formatted

    def _adapt_for_mode(self, text: str, mode: str) -> str:
        """Adapt response for different output modes."""
        if mode == "voice":
            # Remove markdown formatting and emojis for voice
            voice_text = text
            voice_text = voice_text.replace("**", "")
            voice_text = voice_text.replace("*", "")
            voice_text = voice_text.replace("#", "")

            # Remove emojis (simple approach)
            import re

            voice_text = re.sub(r"[^\w\s\.,!?;:()\-]", "", voice_text)

            # Clean up extra whitespace
            voice_text = " ".join(voice_text.split())

            return voice_text
        elif mode == "both":
            # Return both formatted text and voice-adapted version
            return {"text": text, "voice": self._adapt_for_mode(text, "voice")}
        else:
            # Default text mode
            return text

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    def create_summary(self, full_response: str, max_length: int = None) -> str:
        """Create a summary of a longer response."""
        if max_length is None:
            max_length = self.max_summary_length

        if len(full_response) <= max_length:
            return full_response

        # Find a good breaking point (sentence end)
        truncated = full_response[:max_length]
        last_sentence = truncated.rfind(".")
        last_exclaim = truncated.rfind("!")
        last_question = truncated.rfind("?")

        break_point = max(last_sentence, last_exclaim, last_question)

        if break_point > max_length * 0.7:  # At least 70% of desired length
            return truncated[: break_point + 1] + "..."
        else:
            return truncated + "..."

    def format_structured_data(self, data: Dict, title: str = "Data") -> str:
        """Format structured data in a readable way."""
        formatted = f"📋 **{title}**\n\n"

        def format_value(key: str, value: Any, indent: int = 0) -> str:
            prefix = "  " * indent

            if isinstance(value, dict):
                result = f"{prefix}**{key}:**\n"
                for k, v in value.items():
                    result += format_value(k, v, indent + 1)
                return result
            elif isinstance(value, list):
                if not value:
                    return f"{prefix}• {key}: (empty)\n"
                elif len(value) <= 5:
                    return f"{prefix}• {key}: {', '.join(map(str, value))}\n"
                else:
                    return f"{prefix}• {key}: {', '.join(map(str, value[:5]))} ... (+{len(value) - 5} more)\n"
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    return f"{prefix}• {key}: {value:.2f}\n"
                else:
                    return f"{prefix}• {key}: {value:,}\n"
            else:
                return f"{prefix}• {key}: {str(value)}\n"

        for key, value in data.items():
            formatted += format_value(key, value)

        return formatted

    def format_progress_update(
        self, operation: str, current: int, total: int, details: str = ""
    ) -> str:
        """Format progress update message."""
        percentage = (current / total * 100) if total > 0 else 0

        # Create progress bar
        bar_length = 20
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        formatted = f"⏳ **{operation}**\n"
        formatted += f"Progress: [{bar}] {percentage:.1f}% ({current:,}/{total:,})\n"

        if details:
            formatted += f"Current: {details}\n"

        return formatted
