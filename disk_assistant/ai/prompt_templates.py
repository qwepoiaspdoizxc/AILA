from typing import Dict, List, Any, Optional


class PromptTemplates:
    """Collection of prompt templates for AI interactions."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the main system prompt for the AI assistant."""
        return """You are an intelligent local disk assistant AI. Your role is to help users:

1. **File Management**: Find, organize, and manage files on their local computer
2. **Data Analysis**: Analyze CSV, Excel, and other data files with insights
3. **System Optimization**: Suggest file organization and cleanup strategies
4. **Search Assistance**: Help users find files using natural language queries
5. **Productivity**: Provide actionable recommendations for file workflows

Key Guidelines:
- Always be helpful, accurate, and concise
- Focus on practical, actionable advice
- When analyzing data, provide clear insights and visualizations
- Respect user privacy - all processing is local
- If you're uncertain, ask clarifying questions
- Suggest specific actions users can take

Current Context: You have access to the user's file index, search results, and data analysis capabilities."""

    @staticmethod
    def get_file_search_prompt(query: str, results: List[Dict]) -> str:
        """Generate prompt for file search assistance."""
        results_text = ""
        if results:
            results_text = "\nSearch Results:\n"
            for i, result in enumerate(results[:10], 1):
                results_text += f"{i}. {result.get('filename', 'Unknown')} ({result.get('file_type', 'unknown')})\n"
                results_text += f"   Path: {result.get('path', 'Unknown')}\n"
                if result.get("last_modified"):
                    results_text += f"   Modified: {result['last_modified']}\n"
                results_text += "\n"
        else:
            results_text = "\nNo files found matching the search criteria.\n"

        return f"""User is searching for files with query: "{query}"
{results_text}
Please help the user by:
1. Summarizing what was found (or not found)
2. If files were found, highlight the most relevant ones
3. If no files were found, suggest alternative search terms or strategies
4. Provide actionable next steps

Be conversational and helpful in your response."""

    @staticmethod
    def get_data_analysis_prompt(file_info: Dict, analysis_results: Dict) -> str:
        """Generate prompt for data analysis interpretation."""
        return f"""The user has requested analysis of a data file:

File Information:
- Name: {file_info.get("filename", "Unknown")}
- Type: {file_info.get("file_type", "Unknown")}
- Size: {file_info.get("file_size", "Unknown")} bytes

Analysis Results:
{analysis_results}

Please provide:
1. A clear summary of what the data contains
2. Key insights and patterns you notice
3. Data quality observations (missing values, duplicates, etc.)
4. Recommendations for further analysis or data cleaning
5. Suggestions for visualizations that would be helpful

Make your response accessible to non-technical users while being thorough."""

    @staticmethod
    def get_file_organization_prompt(file_stats: Dict) -> str:
        """Generate prompt for file organization suggestions."""
        return f"""Based on the user's file system analysis:

File Statistics:
- Total files indexed: {file_stats.get("total_files", 0)}
- File types: {file_stats.get("file_types", {})}
- Duplicate files: {file_stats.get("duplicates", 0)}
- Large files (>100MB): {file_stats.get("large_files", 0)}
- Recently modified: {file_stats.get("recent_files", 0)}

Please provide:
1. File organization recommendations
2. Suggestions for cleaning up duplicates or large files
3. Folder structure improvements
4. Automation opportunities
5. Best practices for maintaining organization

Focus on practical, actionable advice that will improve the user's file management workflow."""

    @staticmethod
    def get_intent_classification_prompt(query: str) -> str:
        """Generate prompt for classifying user intent."""
        return f"""Analyze this user query and determine the primary intent:

Query: "{query}"

Classify the intent as one of:
- SEARCH: Finding specific files or content
- ANALYZE: Data analysis or file content analysis  
- ORGANIZE: File organization, cleanup, or management
- OPEN: Opening or launching files/applications
- DELETE: Removing or cleaning up files
- HELP: General assistance or how-to questions
- CHAT: General conversation or unclear intent

Respond with just the intent category and a confidence score (0-1).
Format: INTENT_CATEGORY (confidence: 0.XX)"""

    @staticmethod
    def get_error_handling_prompt(error_type: str, error_details: str) -> str:
        """Generate prompt for handling errors gracefully."""
        return f"""An error occurred while processing the user's request:

Error Type: {error_type}
Details: {error_details}

Please provide a helpful response that:
1. Acknowledges the issue in user-friendly terms
2. Suggests possible solutions or workarounds
3. Offers alternative approaches to achieve their goal
4. Maintains a helpful and encouraging tone

Avoid technical jargon and focus on actionable solutions."""

    @staticmethod
    def get_conversation_context_prompt(history: List[Dict], current_query: str) -> str:
        """Generate prompt with conversation context."""
        context = "Previous conversation:\n"
        for exchange in history[-3:]:  # Last 3 exchanges
            context += f"User: {exchange.get('user', '')}\n"
            context += f"Assistant: {exchange.get('assistant', '')}\n\n"

        return f"""{context}Current query: "{current_query}"

Continue the conversation naturally, referencing previous context when relevant. 
Provide helpful and consistent assistance based on the ongoing discussion."""

    @staticmethod
    def get_file_content_summary_prompt(file_path: str, content_preview: str) -> str:
        """Generate prompt for summarizing file contents."""
        return f"""Please analyze and summarize this file:

File: {file_path}
Content Preview:
{content_preview}

Provide:
1. A brief summary of the file's purpose and content
2. Key topics or themes identified
3. Notable features or structure
4. Suggestions for how this file might be used or organized
5. Any quality or formatting observations

Keep the summary concise but informative."""

    @staticmethod
    def get_productivity_tips_prompt(usage_patterns: Dict) -> str:
        """Generate prompt for productivity recommendations."""
        return f"""Based on the user's file access patterns:

Usage Analysis:
- Most accessed file types: {usage_patterns.get("frequent_types", [])}
- Common search terms: {usage_patterns.get("search_terms", [])}
- Peak usage times: {usage_patterns.get("peak_times", [])}
- Frequently accessed folders: {usage_patterns.get("frequent_folders", [])}

Provide personalized productivity tips:
1. Workflow optimization suggestions
2. File organization strategies based on usage
3. Search efficiency tips
4. Automation opportunities
5. Time-saving shortcuts

Tailor recommendations to their specific usage patterns."""


class ResponseFormatter:
    """Format AI responses for different contexts and output modes."""

    @staticmethod
    def format_search_response(ai_response: str, results: List[Dict]) -> str:
        """Format search response with results."""
        if not results:
            return f"🔍 **Search Results**: No files found\n\n{ai_response}"

        formatted = f"🔍 **Search Results**: Found {len(results)} file(s)\n\n"
        formatted += f"{ai_response}\n\n"

        if len(results) <= 5:
            formatted += "📁 **Files Found:**\n"
            for i, result in enumerate(results, 1):
                formatted += f"{i}. {result.get('filename', 'Unknown')}\n"
                formatted += f"   📂 {result.get('path', 'Unknown')}\n"

        return formatted

    @staticmethod
    def format_analysis_response(ai_response: str, charts: List[str] = None) -> str:
        """Format data analysis response."""
        formatted = f"📊 **Data Analysis Complete**\n\n{ai_response}\n\n"

        if charts:
            formatted += "📈 **Generated Visualizations:**\n"
            for chart in charts:
                formatted += f"• {chart}\n"

        return formatted

    @staticmethod
    def format_error_response(ai_response: str, error_type: str) -> str:
        """Format error response."""
        icon = "⚠️" if error_type == "warning" else "❌"
        return f"{icon} **Issue Encountered**\n\n{ai_response}"

    @staticmethod
    def format_help_response(ai_response: str) -> str:
        """Format help response."""
        return f"💡 **Assistant Help**\n\n{ai_response}"
