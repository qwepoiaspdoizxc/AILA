import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from utils.logger import logger

# Try importing spaCy
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not available. Some entity extraction features will be limited."
    )


class EntityExtractor:
    """Extract entities from natural language queries for file management operations."""

    def __init__(self):
        self.nlp = None
        self.file_type_patterns = self._build_file_type_patterns()
        self.size_patterns = self._build_size_patterns()
        self.time_patterns = self._build_time_patterns()
        self.action_patterns = self._build_action_patterns()

        if SPACY_AVAILABLE:
            self._initialize_spacy()

    def _initialize_spacy(self):
        """Initialize spaCy model for advanced NLP."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("EntityExtractor: Loaded spaCy language model")
        except OSError:
            logger.warning(
                "spaCy model not found. Using pattern-based extraction only."
            )
            self.nlp = None

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract all relevant entities from a query."""
        query_lower = query.lower().strip()

        entities = {
            "original_query": query,
            "processed_query": query_lower,
            "file_types": self._extract_file_types(query_lower),
            "file_names": self._extract_file_names(query_lower),
            "file_sizes": self._extract_file_sizes(query_lower),
            "time_constraints": self._extract_time_constraints(query_lower),
            "locations": self._extract_locations(query),
            "actions": self._extract_actions(query_lower),
            "attributes": self._extract_file_attributes(query_lower),
            "operators": self._extract_operators(query_lower),
            "keywords": self._extract_keywords(query_lower),
            "intent": self._determine_intent(query_lower),
        }

        # Use spaCy for additional entity extraction if available
        if self.nlp:
            entities.update(self._extract_spacy_entities(query))

        return entities

    def _extract_file_types(self, query: str) -> List[Dict[str, Any]]:
        """Extract file types and extensions from query."""
        file_types = []

        # Direct extension matches
        extension_pattern = r"\.([\w]{1,6})(?:\s|$|[^\w])"
        extensions = re.findall(extension_pattern, query)
        for ext in extensions:
            file_types.append(
                {"type": "extension", "value": f".{ext.lower()}", "confidence": 0.9}
            )

        # Named file type matches
        for pattern, info in self.file_type_patterns.items():
            if re.search(pattern, query):
                file_types.append(
                    {
                        "type": "category",
                        "value": info["extensions"],
                        "category": info["category"],
                        "confidence": 0.8,
                    }
                )

        return file_types

    def _extract_file_names(self, query: str) -> List[Dict[str, Any]]:
        """Extract specific file names from query."""
        file_names = []

        # Files with extensions
        filename_pattern = (
            r'(?:file|document|named?)\s+["\']?([^\s"\']+\.[a-zA-Z0-9]{1,6})["\']?'
        )
        matches = re.findall(filename_pattern, query, re.IGNORECASE)
        for match in matches:
            file_names.append({"type": "exact_name", "value": match, "confidence": 0.9})

        # Quoted filenames
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_matches = re.findall(quoted_pattern, query)
        for match in quoted_matches:
            if "." in match and len(match.split(".")[-1]) <= 6:
                file_names.append(
                    {"type": "quoted_name", "value": match, "confidence": 0.8}
                )

        return file_names

    def _extract_file_sizes(self, query: str) -> List[Dict[str, Any]]:
        """Extract file size constraints from query."""
        sizes = []

        for pattern, info in self.size_patterns.items():
            matches = re.finditer(pattern, query)
            for match in matches:
                size_value = float(match.group(1)) if match.group(1) else None
                unit = (
                    match.group(2).lower()
                    if len(match.groups()) > 1 and match.group(2)
                    else "b"
                )

                sizes.append(
                    {
                        "type": info["type"],
                        "value": size_value,
                        "unit": unit,
                        "operator": info.get("operator", "equal"),
                        "bytes": self._convert_to_bytes(size_value, unit)
                        if size_value
                        else None,
                        "confidence": 0.8,
                    }
                )

        return sizes

    def _extract_time_constraints(self, query: str) -> List[Dict[str, Any]]:
        """Extract time-based constraints from query."""
        time_constraints = []

        for pattern, info in self.time_patterns.items():
            if re.search(pattern, query):
                constraint = {
                    "type": info["type"],
                    "period": info["period"],
                    "confidence": 0.8,
                }

                # Calculate actual datetime range
                if info["type"] == "relative":
                    constraint["datetime_range"] = self._calculate_time_range(
                        info["period"]
                    )

                time_constraints.append(constraint)

        # Extract specific dates
        date_patterns = [
            r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
            r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
            r"(\d{1,2})-(\d{1,2})-(\d{4})",  # DD-MM-YYYY
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                try:
                    # Assume first pattern is MM/DD/YYYY
                    date_obj = datetime(int(match[2]), int(match[0]), int(match[1]))
                    time_constraints.append(
                        {"type": "specific_date", "date": date_obj, "confidence": 0.9}
                    )
                except ValueError:
                    continue

        return time_constraints

    def _extract_locations(self, query: str) -> List[Dict[str, Any]]:
        """Extract file location/path information."""
        locations = []

        # Windows path patterns
        path_patterns = [
            r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\?)*',  # C:\path\to\folder
            r'\\\\[^\\/:*?"<>|\r\n]+\\[^\\/:*?"<>|\r\n\\]*',  # \\server\share
        ]

        for pattern in path_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if os.path.exists(match):
                    locations.append(
                        {
                            "type": "absolute_path",
                            "value": match,
                            "exists": True,
                            "confidence": 0.9,
                        }
                    )
                else:
                    locations.append(
                        {
                            "type": "potential_path",
                            "value": match,
                            "exists": False,
                            "confidence": 0.6,
                        }
                    )

        # Common directory names
        common_dirs = [
            "desktop",
            "documents",
            "downloads",
            "pictures",
            "music",
            "videos",
            "temp",
            "temporary",
            "recycle bin",
            "trash",
        ]

        for dir_name in common_dirs:
            if dir_name in query.lower():
                locations.append(
                    {"type": "common_directory", "value": dir_name, "confidence": 0.7}
                )

        return locations

    def _extract_actions(self, query: str) -> List[Dict[str, Any]]:
        """Extract intended actions from query."""
        actions = []

        for pattern, info in self.action_patterns.items():
            if re.search(pattern, query):
                actions.append(
                    {
                        "action": info["action"],
                        "category": info["category"],
                        "confidence": 0.8,
                    }
                )

        return actions

    def _extract_file_attributes(self, query: str) -> List[Dict[str, Any]]:
        """Extract file attributes and properties."""
        attributes = []

        # Attribute patterns
        attribute_patterns = {
            r"\b(?:empty|blank)\b": {
                "attribute": "size",
                "value": 0,
                "operator": "equal",
            },
            r"\b(?:hidden)\b": {"attribute": "hidden", "value": True},
            r"\b(?:read-?only|readonly)\b": {"attribute": "readonly", "value": True},
            r"\b(?:executable?)\b": {"attribute": "executable", "value": True},
            r"\b(?:corrupted?|damaged?)\b": {"attribute": "corrupted", "value": True},
            r"\b(?:duplicate|duplicated)\b": {"attribute": "duplicate", "value": True},
        }

        for pattern, attr_info in attribute_patterns.items():
            if re.search(pattern, query):
                attributes.append(
                    {
                        "type": "property",
                        "attribute": attr_info["attribute"],
                        "value": attr_info["value"],
                        "operator": attr_info.get("operator", "equal"),
                        "confidence": 0.7,
                    }
                )

        return attributes

    def _extract_operators(self, query: str) -> List[str]:
        """Extract logical operators from query."""
        operators = []

        operator_patterns = {
            r"\b(?:and|&)\b": "AND",
            r"\b(?:or|\|)\b": "OR",
            r"\b(?:not|!|-)\b": "NOT",
            r"\b(?:greater than|larger than|bigger than|>)\b": "GT",
            r"\b(?:less than|smaller than|<)\b": "LT",
            r"\b(?:equal to|equals?|=)\b": "EQ",
        }

        for pattern, op in operator_patterns.items():
            if re.search(pattern, query):
                operators.append(op)

        return list(set(operators))  # Remove duplicates

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords for search."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "find",
            "search",
            "show",
            "get",
            "file",
            "files",
        }

        words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return list(set(keywords))

    def _determine_intent(self, query: str) -> Dict[str, Any]:
        """Determine the primary intent of the query."""
        intent_patterns = {
            "search": r"\b(?:find|search|look|locate|where|show|list)\b",
            "open": r"\b(?:open|launch|start|run|execute)\b",
            "delete": r"\b(?:delete|remove|trash|erase)\b",
            "copy": r"\b(?:copy|duplicate)\b",
            "move": r"\b(?:move|transfer|relocate)\b",
            "analyze": r"\b(?:analyze|statistics|chart|graph|report)\b",
            "organize": r"\b(?:organize|sort|arrange|group)\b",
            "info": r"\b(?:info|information|details|properties|metadata)\b",
        }

        intents = []
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, query):
                intents.append({"intent": intent, "confidence": 0.8})

        # Return primary intent or default to search
        if intents:
            return max(intents, key=lambda x: x["confidence"])
        else:
            return {"intent": "search", "confidence": 0.5}

    def _extract_spacy_entities(self, query: str) -> Dict[str, Any]:
        """Use spaCy for additional entity extraction."""
        if not self.nlp:
            return {}

        doc = self.nlp(query)

        spacy_entities = {"named_entities": [], "pos_tags": [], "dependencies": []}

        # Named entities
        for ent in doc.ents:
            spacy_entities["named_entities"].append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )

        # Part-of-speech tags for important words
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]:
                spacy_entities["pos_tags"].append(
                    {
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "tag": token.tag_,
                    }
                )

        return spacy_entities

    def _build_file_type_patterns(self) -> Dict[str, Dict]:
        """Build patterns for file type recognition."""
        return {
            r"\b(?:document|doc|word|text)\b": {
                "category": "document",
                "extensions": [".doc", ".docx", ".txt", ".rtf", ".odt"],
            },
            r"\b(?:spreadsheet|excel|xls)\b": {
                "category": "spreadsheet",
                "extensions": [".xls", ".xlsx", ".csv"],
            },
            r"\b(?:presentation|powerpoint|ppt)\b": {
                "category": "presentation",
                "extensions": [".ppt", ".pptx"],
            },
            r"\b(?:image|picture|photo)\b": {
                "category": "image",
                "extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
            },
            r"\b(?:audio|music|sound)\b": {
                "category": "audio",
                "extensions": [".mp3", ".wav", ".wma", ".aac", ".flac"],
            },
            r"\b(?:video|movie|film)\b": {
                "category": "video",
                "extensions": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
            },
            r"\b(?:archive|zip|compressed)\b": {
                "category": "archive",
                "extensions": [".zip", ".rar", ".7z", ".tar", ".gz"],
            },
            r"\b(?:code|programming|script)\b": {
                "category": "code",
                "extensions": [".py", ".js", ".java", ".cpp", ".c", ".html", ".css"],
            },
        }

    def _build_size_patterns(self) -> Dict[str, Dict]:
        """Build patterns for file size recognition."""
        return {
            r"\b(?:larger than|bigger than|greater than|over|above)\s+(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb|bytes?|b)?\b": {
                "type": "size_constraint",
                "operator": "greater_than",
            },
            r"\b(?:smaller than|less than|under|below)\s+(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb|bytes?|b)?\b": {
                "type": "size_constraint",
                "operator": "less_than",
            },
            r"\b(?:exactly|equal to)?\s*(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb|bytes?|b)\b": {
                "type": "size_constraint",
                "operator": "equal",
            },
        }

    def _build_time_patterns(self) -> Dict[str, Dict]:
        """Build patterns for time-based constraints."""
        return {
            r"\b(?:today|this day)\b": {"type": "relative", "period": "today"},
            r"\b(?:yesterday)\b": {"type": "relative", "period": "yesterday"},
            r"\b(?:this week|past week|last week)\b": {
                "type": "relative",
                "period": "week",
            },
            r"\b(?:this month|past month|last month)\b": {
                "type": "relative",
                "period": "month",
            },
            r"\b(?:this year|past year|last year)\b": {
                "type": "relative",
                "period": "year",
            },
            r"\b(?:recent|recently)\b": {"type": "relative", "period": "recent"},
            r"\b(?:old|older)\b": {"type": "relative", "period": "old"},
        }

    def _build_action_patterns(self) -> Dict[str, Dict]:
        """Build patterns for action recognition."""
        return {
            r"\b(?:find|search|look|locate|show|list|get)\b": {
                "action": "search",
                "category": "query",
            },
            r"\b(?:open|launch|start|run|execute)\b": {
                "action": "open",
                "category": "execute",
            },
            r"\b(?:delete|remove|trash|erase)\b": {
                "action": "delete",
                "category": "modify",
            },
            r"\b(?:copy|duplicate)\b": {"action": "copy", "category": "modify"},
            r"\b(?:move|transfer|relocate)\b": {"action": "move", "category": "modify"},
            r"\b(?:analyze|statistics|chart|graph|report)\b": {
                "action": "analyze",
                "category": "analyze",
            },
        }

    def _convert_to_bytes(self, value: float, unit: str) -> int:
        """Convert file size to bytes."""
        unit = unit.lower().replace("s", "").replace("byte", "b")

        multipliers = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}

        return int(value * multipliers.get(unit, 1))

    def _calculate_time_range(self, period: str) -> Tuple[datetime, datetime]:
        """Calculate datetime range for relative time periods."""
        now = datetime.now()

        if period == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period == "yesterday":
            start = (now - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = start.replace(hour=23, minute=59, second=59)
        elif period == "week":
            start = now - timedelta(days=7)
            end = now
        elif period == "month":
            start = now - timedelta(days=30)
            end = now
        elif period == "year":
            start = now - timedelta(days=365)
            end = now
        elif period == "recent":
            start = now - timedelta(days=3)
            end = now
        else:  # 'old'
            start = datetime(1970, 1, 1)
            end = now - timedelta(days=365)

        return (start, end)

    def create_search_filters(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extracted entities into search filters."""
        filters = {
            "file_types": [],
            "size_constraints": [],
            "time_constraints": [],
            "name_patterns": [],
            "location_filters": [],
            "attribute_filters": [],
            "keywords": entities.get("keywords", []),
        }

        # Process file types
        for file_type in entities.get("file_types", []):
            if file_type["type"] == "extension":
                filters["file_types"].append(file_type["value"])
            elif file_type["type"] == "category":
                filters["file_types"].extend(file_type["value"])

        # Process size constraints
        for size in entities.get("file_sizes", []):
            if size.get("bytes"):
                filters["size_constraints"].append(
                    {"operator": size["operator"], "bytes": size["bytes"]}
                )

        # Process time constraints
        for time_constraint in entities.get("time_constraints", []):
            if "datetime_range" in time_constraint:
                filters["time_constraints"].append(time_constraint["datetime_range"])

        # Process file names
        for name in entities.get("file_names", []):
            filters["name_patterns"].append(name["value"])

        # Process locations
        for location in entities.get("locations", []):
            filters["location_filters"].append(location["value"])

        # Process attributes
        for attr in entities.get("attributes", []):
            filters["attribute_filters"].append(
                {
                    "attribute": attr["attribute"],
                    "value": attr["value"],
                    "operator": attr.get("operator", "equal"),
                }
            )

        return filters
