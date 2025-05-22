import sys
import logging
from tkinter import messagebox

# Third-party imports
try:
    import spacy
except ImportError:
    raise ImportError("spacy module not found. Please install it with: pip install spacy")

logger = logging.getLogger("DiskAssistant")


class QueryProcessor:
    """Process natural language queries using spaCy."""

    def __init__(self):
        try:
            # Load the small English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy language model")
        except OSError:
            # If model not found, try to download it
            logger.warning("spaCy model not found, attempting download...")
            try:
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                )
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Downloaded spaCy model successfully")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                messagebox.showerror(
                    "Error",
                    "Could not load language model. Please run:\n\n"
                    "python -m spacy download en_core_web_sm",
                )
                self.nlp = None

    def process_query(self, text):
        """Process a query and extract intent and entities."""
        if not self.nlp:
            # Fallback if nlp not available
            return {
                "intent": "search",
                "file_type": None,
                "location": None,
                "action": None,
                "time_period": None,
                "original_query": text,
            }

        doc = self.nlp(text.lower())

        # Extract key information
        result = {
            "intent": "search",  # Default intent
            "file_type": None,
            "location": None,
            "action": None,
            "time_period": None,
            "original_query": text,
        }

        # Extract verbs to determine action
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        # Determine intent based on verbs
        if any(v in ["find", "search", "look", "locate"] for v in verbs):
            result["intent"] = "search"
        elif any(v in ["open", "launch", "start", "run", "execute"] for v in verbs):
            result["intent"] = "open"
        elif any(v in ["delete", "remove", "trash"] for v in verbs):
            result["intent"] = "delete"

        # Extract file types
        file_types = [
            ".txt", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
            ".ppt", ".pptx", ".jpg", ".png", ".mp3", ".mp4",
        ]
        
        for token in doc:
            # Look for file extensions
            if token.text.startswith(".") and token.text in file_types:
                result["file_type"] = token.text

            # Look for file type names
            if token.text in [
                "text", "document", "word", "excel", "spreadsheet",
                "powerpoint", "presentation", "image", "audio", "video",
            ]:
                result["file_type"] = self._map_file_type_name(token.text)

        # Extract potential locations
        for ent in doc.ents:
            if ent.label_ == "LOC":
                result["location"] = ent.text

        # Extract time periods
        time_indicators = ["today", "yesterday", "last week", "last month", "recent"]
        for indicator in time_indicators:
            if indicator in text.lower():
                result["time_period"] = indicator

        return result

    def _map_file_type_name(self, name):
        """Map file type names to extensions."""
        mapping = {
            "text": ".txt",
            "document": ".docx",
            "word": ".docx",
            "excel": ".xlsx",
            "spreadsheet": ".xlsx",
            "powerpoint": ".pptx",
            "presentation": ".pptx",
            "image": [".jpg", ".png"],
            "audio": ".mp3",
            "video": ".mp4",
        }
        return mapping.get(name)

    def generate_search_query(self, processed_query):
        """Convert processed query to a search string for Whoosh."""
        if not self.nlp:
            # Simple fallback if nlp not available
            return processed_query["original_query"]

        query_parts = []

        # Add the original query terms (excluding stop words)
        doc = self.nlp(processed_query["original_query"])
        content_words = [
            token.text for token in doc if not token.is_stop and token.is_alpha
        ]
        if content_words:
            query_parts.append(" ".join(content_words))

        # Add file type if specified
        if processed_query["file_type"]:
            if isinstance(processed_query["file_type"], list):
                file_type_query = " OR ".join(
                    [f'file_type:"{ft}"' for ft in processed_query["file_type"]]
                )
                query_parts.append(f"({file_type_query})")
            else:
                query_parts.append(f'file_type:"{processed_query["file_type"]}"')

        # Combine parts with AND
        if query_parts:
            return " AND ".join(query_parts)
        else:
            return processed_query["original_query"]