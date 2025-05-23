import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from config import CONFIG
from utils.logger import logger


class IntentType(Enum):
    """Enumeration of supported intent types."""

    # File operations
    SEARCH_FILES = "search_files"
    OPEN_FILE = "open_file"
    DELETE_FILE = "delete_file"
    RENAME_FILE = "rename_file"
    COPY_FILE = "copy_file"
    MOVE_FILE = "move_file"

    # Data analysis
    ANALYZE_DATA = "analyze_data"
    CREATE_CHART = "create_chart"
    STATISTICS = "statistics"
    COMPARE_FILES = "compare_files"

    # Organization
    ORGANIZE_FILES = "organize_files"
    CLEAN_DUPLICATES = "clean_duplicates"
    BACKUP_FILES = "backup_files"

    # Information
    FILE_INFO = "file_info"
    SYSTEM_INFO = "system_info"
    HELP = "help"
    LIST_FILES = "list_files"

    # Voice/Audio
    RECORD_AUDIO = "record_audio"
    TRANSCRIBE_AUDIO = "transcribe_audio"

    # General
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: IntentType
    confidence: float
    parameters: Dict[str, Any]
    suggested_action: str
    entities: Dict[str, List[str]]
    reasoning: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["intent"] = self.intent.value
        return result


class EntityExtractor:
    """Extract entities from user queries."""

    def __init__(self):
        # File extension patterns
        self.file_extensions = set(CONFIG.get("file_extensions", []))

        # Common file operation patterns
        self.patterns = {
            "file_path": re.compile(
                r'["\']?([A-Za-z]:\\[^"\']*|\/[^"\']*|[^\\\/\s"\']+\.[a-zA-Z0-9]+)["\']?'
            ),
            "file_name": re.compile(r'["\']?([^\\\/\s"\']+\.[a-zA-Z0-9]+)["\']?'),
            "size_value": re.compile(
                r"(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb|bytes?)", re.IGNORECASE
            ),
            "date_range": re.compile(
                r"(today|yesterday|last\s+week|last\s+month|last\s+year|\d{1,2}\/\d{1,2}\/\d{2,4})",
                re.IGNORECASE,
            ),
            "number": re.compile(r"\b(\d+)\b"),
        }

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query text."""
        entities = {
            "file_paths": [],
            "file_names": [],
            "file_extensions": [],
            "sizes": [],
            "dates": [],
            "numbers": [],
            "keywords": [],
        }

        # Extract file paths and names
        for match in self.patterns["file_path"].finditer(query):
            path = match.group(1)
            entities["file_paths"].append(path)

            # Extract filename from path
            filename = Path(path).name
            if filename:
                entities["file_names"].append(filename)

        # Extract standalone filenames
        for match in self.patterns["file_name"].finditer(query):
            filename = match.group(1)
            if filename not in entities["file_names"]:
                entities["file_names"].append(filename)

        # Extract file extensions
        query_lower = query.lower()
        for ext in self.file_extensions:
            if ext.lower() in query_lower:
                entities["file_extensions"].append(ext)

        # Extract sizes
        for match in self.patterns["size_value"].finditer(query):
            size_str = match.group(0)
            entities["sizes"].append(size_str)

        # Extract dates
        for match in self.patterns["date_range"].finditer(query):
            date_str = match.group(1)
            entities["dates"].append(date_str)

        # Extract numbers
        for match in self.patterns["number"].finditer(query):
            number = int(match.group(1))
            entities["numbers"].append(number)

        # Extract keywords (remove common words)
        keywords = self._extract_keywords(query)
        entities["keywords"] = keywords

        return entities

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Common stop words to ignore
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
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
        }

        # Clean and split query
        words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
        keywords = [word for word in words if word not in stop_words]

        return list(set(keywords))  # Remove duplicates


class RuleBasedClassifier:
    """Rule-based intent classification."""

    def __init__(self):
        self.intent_patterns = {
            IntentType.SEARCH_FILES: [
                r"\b(find|search|look\s+for|locate|where\s+is)\b.*\b(file|files|document|documents)\b",
                r"\b(show\s+me|list|display)\b.*\b(file|files)\b.*\b(containing|with|named)\b",
                r"\b(search|find)\b(?!.*delete|remove)",
            ],
            IntentType.OPEN_FILE: [
                r"\b(open|launch|start|run|execute)\b",
                r"\b(show|display|view)\b.*\b(file|document)\b",
            ],
            IntentType.DELETE_FILE: [
                r"\b(delete|remove|erase|trash)\b.*\b(file|files|document)\b",
                r"\b(get\s+rid\s+of|dispose\s+of)\b.*\b(file|files)\b",
            ],
            IntentType.RENAME_FILE: [
                r"\b(rename|change\s+name)\b.*\b(file|document)\b",
                r"\b(call|name)\b.*\b(instead|something\s+else)\b",
            ],
            IntentType.COPY_FILE: [
                r"\b(copy|duplicate|clone)\b.*\b(file|files|document)\b",
                r"\b(make\s+a\s+copy)\b",
            ],
            IntentType.MOVE_FILE: [
                r"\b(move|relocate|transfer)\b.*\b(file|files|document)\b",
                r"\b(put|place)\b.*\b(in|into|to)\b.*\b(folder|directory)\b",
            ],
            IntentType.ANALYZE_DATA: [
                r"\b(analyze|analyse|examine|study)\b.*\b(data|content|file)\b",
                r"\b(what\s+is\s+in|summarize|summary)\b.*\b(file|document)\b",
                r"\b(insights|patterns|trends)\b",
            ],
            IntentType.CREATE_CHART: [
                r"\b(create|make|generate|build)\b.*\b(chart|graph|plot|visualization)\b",
                r"\b(visualize|show\s+graphically)\b",
                r"\b(bar\s+chart|line\s+graph|pie\s+chart|histogram)\b",
            ],
            IntentType.STATISTICS: [
                r"\b(statistics|stats|count|total|average|mean|median)\b",
                r"\b(how\s+many|number\s+of)\b.*\b(files|documents)\b",
                r"\b(size|space|usage)\b.*\b(disk|storage|directory)\b",
            ],
            IntentType.COMPARE_FILES: [
                r"\b(compare|difference|diff|similar|same)\b.*\b(file|files|document)\b",
                r"\b(what\s+changed|modifications|updates)\b",
            ],
            IntentType.ORGANIZE_FILES: [
                r"\b(organize|sort|arrange|group|categorize)\b.*\b(file|files|document)\b",
                r"\b(clean\s+up|tidy|structure)\b.*\b(folder|directory)\b",
            ],
            IntentType.CLEAN_DUPLICATES: [
                r"\b(duplicate|duplicates|same\s+file|identical)\b.*\b(remove|delete|clean)\b",
                r"\b(find|remove)\b.*\b(duplicate|duplicates)\b",
            ],
            IntentType.BACKUP_FILES: [
                r"\b(backup|back\s+up|save\s+copy)\b.*\b(file|files|document)\b",
                r"\b(archive|preserve)\b",
            ],
            IntentType.FILE_INFO: [
                r"\b(information|info|details|properties|metadata)\b.*\b(file|document)\b",
                r"\b(when\s+was|created|modified|size|type)\b.*\b(file|document)\b",
            ],
            IntentType.SYSTEM_INFO: [
                r"\b(system|computer|disk|storage|memory)\b.*\b(information|info|status)\b",
                r"\b(free\s+space|available\s+space|disk\s+usage)\b",
            ],
            IntentType.LIST_FILES: [
                r"\b(list|show|display)\b.*\b(file|files|document|documents)\b(?!.*search|find)",
                r"\b(what\s+files|which\s+files)\b.*\b(in|inside)\b",
            ],
            IntentType.RECORD_AUDIO: [
                r"\b(record|capture)\b.*\b(audio|sound|voice)\b",
                r"\b(start\s+recording|begin\s+recording)\b",
            ],
            IntentType.TRANSCRIBE_AUDIO: [
                r"\b(transcribe|convert\s+to\s+text)\b.*\b(audio|recording|voice)\b",
                r"\b(speech\s+to\s+text|voice\s+to\s+text)\b",
            ],
            IntentType.HELP: [
                r"\b(help|assistance|support|how\s+to|what\s+can)\b",
                r"\b(commands|options|features|capabilities)\b",
            ],
        }

    def classify(self, query: str) -> Tuple[IntentType, float, str]:
        """Classify intent using rule-based patterns."""
        query_lower = query.lower()
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        best_reasoning = "No matching patterns found"

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(
                        pattern, query_lower
                    )

                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence
                        best_reasoning = f"Matched pattern: {pattern}"

        # If no specific intent found but query seems file-related, default to general query
        if best_intent == IntentType.UNKNOWN:
            if any(
                word in query_lower
                for word in ["file", "document", "folder", "directory"]
            ):
                best_intent = IntentType.GENERAL_QUERY
                best_confidence = 0.3
                best_reasoning = "Contains file-related keywords"

        return best_intent, best_confidence, best_reasoning

    def _calculate_pattern_confidence(self, pattern: str, query: str) -> float:
        """Calculate confidence score for pattern match."""
        # Base confidence for any match
        confidence = 0.6

        # Increase confidence for longer patterns (more specific)
        confidence += min(len(pattern) / 100, 0.2)

        # Increase confidence for multiple keyword matches
        keywords = re.findall(r"\b\w+\b", pattern.replace("\\b", "").replace(".*", ""))
        matches = sum(1 for keyword in keywords if keyword in query)
        confidence += (matches / len(keywords)) * 0.2

        return min(confidence, 1.0)


class IntentClassifier:
    """Main intent classification system."""

    def __init__(self, model_manager=None):
        self.entity_extractor = EntityExtractor()
        self.rule_classifier = RuleBasedClassifier()
        self.model_manager = model_manager
        self.classification_history = []
        self.cache = {}
        self.cache_size_limit = 100

        logger.info("Intent classifier initialized")

    def classify_intent(self, query: str, use_ai: bool = True) -> IntentResult:
        """Classify user intent from query."""
        # Check cache first
        if query in self.cache:
            logger.debug(f"Using cached intent classification for: {query[:50]}...")
            return self.cache[query]

        # Extract entities
        entities = self.entity_extractor.extract_entities(query)

        # Rule-based classification
        rule_intent, rule_confidence, rule_reasoning = self.rule_classifier.classify(
            query
        )

        # AI-powered classification (if available and requested)
        ai_intent = None
        ai_confidence = 0.0
        ai_reasoning = None

        if use_ai and self.model_manager and self.model_manager.get_current_model():
            try:
                ai_result = self._classify_with_ai(query, entities)
                if ai_result:
                    ai_intent, ai_confidence, ai_reasoning = ai_result
            except Exception as e:
                logger.debug(f"AI classification failed: {e}")

        # Combine results
        final_intent, final_confidence, final_reasoning = self._combine_classifications(
            (rule_intent, rule_confidence, rule_reasoning),
            (ai_intent, ai_confidence, ai_reasoning) if ai_intent else None,
        )

        # Generate parameters and suggested action
        parameters = self._extract_parameters(query, entities, final_intent)
        suggested_action = self._get_suggested_action(final_intent, parameters)

        # Create result
        result = IntentResult(
            intent=final_intent,
            confidence=final_confidence,
            parameters=parameters,
            suggested_action=suggested_action,
            entities=entities,
            reasoning=final_reasoning,
        )

        # Cache result
        self._cache_result(query, result)

        # Store in history
        self.classification_history.append(result)
        if len(self.classification_history) > 50:
            self.classification_history = self.classification_history[-50:]

        logger.debug(
            f"Classified intent: {final_intent.value} (confidence: {final_confidence:.2f})"
        )
        return result

    def _classify_with_ai(
        self, query: str, entities: Dict[str, List[str]]
    ) -> Optional[Tuple[IntentType, float, str]]:
        """Use AI model for intent classification."""
        if not self.model_manager:
            return None

        # Prepare prompt for AI classification
        intent_types = [intent.value for intent in IntentType]

        prompt = f"""Classify the following user query into one of these intents:
{", ".join(intent_types)}

User Query: "{query}"

Available entities in the query:
{json.dumps(entities, indent=2)}

Respond with a JSON object containing:
- intent: one of the listed intent types
- confidence: float between 0.0 and 1.0
- reasoning: brief explanation

Response:"""

        try:
            response = self.model_manager.generate_text(prompt, max_tokens=200)

            # Try to parse JSON response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                intent_str = result.get("intent", "unknown")
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "AI classification")

                # Convert intent string to enum
                try:
                    intent = IntentType(intent_str)
                except ValueError:
                    intent = IntentType.UNKNOWN

                return intent, confidence, reasoning

        except Exception as e:
            logger.debug(f"Error parsing AI classification response: {e}")

        return None

    def _combine_classifications(
        self,
        rule_result: Tuple[IntentType, float, str],
        ai_result: Optional[Tuple[IntentType, float, str]],
    ) -> Tuple[IntentType, float, str]:
        """Combine rule-based and AI classification results."""
        rule_intent, rule_confidence, rule_reasoning = rule_result

        if not ai_result:
            return rule_intent, rule_confidence, rule_reasoning

        ai_intent, ai_confidence, ai_reasoning = ai_result

        # If both agree, increase confidence
        if rule_intent == ai_intent:
            combined_confidence = min((rule_confidence + ai_confidence) / 2 + 0.1, 1.0)
            combined_reasoning = f"Rule-based and AI agree: {rule_reasoning}"
            return rule_intent, combined_confidence, combined_reasoning

        # If they disagree, use the one with higher confidence
        if ai_confidence > rule_confidence:
            return ai_intent, ai_confidence, f"AI override: {ai_reasoning}"
        else:
            return rule_intent, rule_confidence, f"Rule-based chosen: {rule_reasoning}"

    def _extract_parameters(
        self, query: str, entities: Dict[str, List[str]], intent: IntentType
    ) -> Dict[str, Any]:
        """Extract relevant parameters based on intent and entities."""
        parameters = {"query": query}

        # Add common parameters
        if entities["file_paths"]:
            parameters["file_paths"] = entities["file_paths"]
        if entities["file_names"]:
            parameters["file_names"] = entities["file_names"]
        if entities["file_extensions"]:
            parameters["file_extensions"] = entities["file_extensions"]
        if entities["keywords"]:
            parameters["keywords"] = entities["keywords"]

        # Intent-specific parameters
        if intent in [IntentType.SEARCH_FILES, IntentType.LIST_FILES]:
            parameters["search_terms"] = entities["keywords"]
            if entities["sizes"]:
                parameters["size_filters"] = entities["sizes"]
            if entities["dates"]:
                parameters["date_filters"] = entities["dates"]

        elif intent == IntentType.STATISTICS:
            if entities["numbers"]:
                parameters["limit"] = (
                    entities["numbers"][0] if entities["numbers"] else None
                )

        elif intent in [IntentType.COPY_FILE, IntentType.MOVE_FILE]:
            # Try to identify source and destination
            if len(entities["file_paths"]) >= 2:
                parameters["source"] = entities["file_paths"][0]
                parameters["destination"] = entities["file_paths"][1]

        elif intent == IntentType.CREATE_CHART:
            # Identify chart type from query
            chart_types = ["bar", "line", "pie", "scatter", "histogram"]
            query_lower = query.lower()
            for chart_type in chart_types:
                if chart_type in query_lower:
                    parameters["chart_type"] = chart_type
                    break

        return parameters

    def _get_suggested_action(
        self, intent: IntentType, parameters: Dict[str, Any]
    ) -> str:
        """Get suggested action based on intent and parameters."""
        action_templates = {
            IntentType.SEARCH_FILES: "Search for files matching criteria",
            IntentType.OPEN_FILE: "Open the specified file",
            IntentType.DELETE_FILE: "Delete the specified file(s)",
            IntentType.RENAME_FILE: "Rename the specified file",
            IntentType.COPY_FILE: "Copy file to destination",
            IntentType.MOVE_FILE: "Move file to destination",
            IntentType.ANALYZE_DATA: "Analyze the data file content",
            IntentType.CREATE_CHART: "Generate visualization from data",
            IntentType.STATISTICS: "Calculate file statistics",
            IntentType.COMPARE_FILES: "Compare the specified files",
            IntentType.ORGANIZE_FILES: "Organize files by criteria",
            IntentType.CLEAN_DUPLICATES: "Find and remove duplicate files",
            IntentType.BACKUP_FILES: "Create backup of files",
            IntentType.FILE_INFO: "Display file information",
            IntentType.SYSTEM_INFO: "Show system information",
            IntentType.LIST_FILES: "List files in directory",
            IntentType.RECORD_AUDIO: "Start audio recording",
            IntentType.TRANSCRIBE_AUDIO: "Transcribe audio to text",
            IntentType.HELP: "Provide help information",
            IntentType.GENERAL_QUERY: "Process general query",
            IntentType.UNKNOWN: "Clarify user request",
        }

        base_action = action_templates.get(intent, "Process user request")

        # Customize action based on parameters
        if "file_names" in parameters and parameters["file_names"]:
            file_name = parameters["file_names"][0]
            base_action = base_action.replace("file", f"file '{file_name}'")

        return base_action

    def _cache_result(self, query: str, result: IntentResult):
        """Cache classification result."""
        if len(self.cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[query] = result

    def get_classification_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent classification history."""
        return [result.to_dict() for result in self.classification_history[-limit:]]

    def clear_cache(self):
        """Clear classification cache."""
        self.cache.clear()
        logger.info("Intent classification cache cleared")

    def clear_history(self):
        """Clear classification history."""
        self.classification_history.clear()
        logger.info("Intent classification history cleared")

    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get statistics about classified intents."""
        if not self.classification_history:
            return {"total_classifications": 0}

        intent_counts = {}
        confidence_scores = []

        for result in self.classification_history:
            intent_name = result.intent.value
            intent_counts[intent_name] = intent_counts.get(intent_name, 0) + 1
            confidence_scores.append(result.confidence)

        return {
            "total_classifications": len(self.classification_history),
            "intent_distribution": intent_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "cache_size": len(self.cache),
        }
