import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import threading
import time

from config import CONFIG
from utils.logger import logger
from ai_assistant import AIAssistant
from data_analyzer import DataAnalyzer


class InsightEngine:
    """Generate intelligent insights about files, usage patterns, and data trends."""

    def __init__(self):
        self.ai_assistant = AIAssistant()
        self.data_analyzer = DataAnalyzer()
        self.db_path = CONFIG.get("db_path", "file_metadata.db")
        self.insight_cache = {}
        self.last_analysis_time = {}
        self.cache_duration = 3600  # 1 hour
        self.insight_history = []
        self.max_history_items = 100

    def initialize(self) -> bool:
        """Initialize the insight engine."""
        try:
            # Initialize AI assistant
            if not self.ai_assistant.initialize():
                logger.warning(
                    "AI assistant failed to initialize - insights will be limited"
                )

            # Ensure database exists
            if not os.path.exists(self.db_path):
                logger.error(f"Database not found: {self.db_path}")
                return False

            logger.info("Insight engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing insight engine: {e}")
            return False

    def generate_comprehensive_insights(
        self, query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive insights about the file system."""
        insights = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "file_system_insights": self._analyze_file_system_patterns(),
            "usage_insights": self._analyze_usage_patterns(),
            "storage_insights": self._analyze_storage_patterns(),
            "data_quality_insights": self._analyze_data_quality(),
            "security_insights": self._analyze_security_patterns(),
            "performance_insights": self._analyze_performance_patterns(),
            "ai_recommendations": self._generate_ai_recommendations(query),
            "trend_analysis": self._analyze_trends(),
            "anomaly_detection": self._detect_anomalies(),
        }

        # Add to history
        self._add_to_history(insights)

        return insights

    def _analyze_file_system_patterns(self) -> Dict[str, Any]:
        """Analyze file system organization and patterns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get file distribution by type
                cursor.execute("""
                    SELECT file_extension, COUNT(*) as count, 
                           SUM(file_size) as total_size,
                           AVG(file_size) as avg_size
                    FROM file_metadata 
                    GROUP BY file_extension 
                    ORDER BY count DESC
                """)
                file_types = cursor.fetchall()

                # Directory depth analysis
                cursor.execute("""
                    SELECT file_path, 
                           LENGTH(file_path) - LENGTH(REPLACE(file_path, '/', '')) as depth
                    FROM file_metadata
                """)
                depth_data = cursor.fetchall()

                # File age distribution
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN (julianday('now') - julianday(last_modified)) < 7 THEN 'This Week'
                            WHEN (julianday('now') - julianday(last_modified)) < 30 THEN 'This Month'
                            WHEN (julianday('now') - julianday(last_modified)) < 90 THEN 'Last 3 Months'
                            WHEN (julianday('now') - julianday(last_modified)) < 365 THEN 'This Year'
                            ELSE 'Older'
                        END as age_group,
                        COUNT(*) as count,
                        SUM(file_size) as total_size
                    FROM file_metadata
                    GROUP BY age_group
                """)
                age_distribution = cursor.fetchall()

                return {
                    "file_type_distribution": [
                        {
                            "extension": row[0] or "No Extension",
                            "count": row[1],
                            "total_size": row[2],
                            "average_size": row[3],
                        }
                        for row in file_types[:20]  # Top 20 types
                    ],
                    "directory_depth_stats": {
                        "average_depth": sum(row[1] for row in depth_data)
                        / len(depth_data)
                        if depth_data
                        else 0,
                        "max_depth": max(row[1] for row in depth_data)
                        if depth_data
                        else 0,
                        "files_in_deep_paths": len(
                            [row for row in depth_data if row[1] > 5]
                        ),
                    },
                    "file_age_distribution": [
                        {"age_group": row[0], "count": row[1], "total_size": row[2]}
                        for row in age_distribution
                    ],
                }
        except Exception as e:
            logger.error(f"Error analyzing file system patterns: {e}")
            return {"error": str(e)}

    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze file usage and access patterns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Most accessed files (if access_count exists)
                cursor.execute("""
                    SELECT name, column_name FROM pragma_table_info('file_metadata')
                """)
                columns = [row[1] for row in cursor.fetchall()]

                insights = {}

                if "access_count" in columns:
                    cursor.execute("""
                        SELECT filename, file_path, access_count, last_accessed
                        FROM file_metadata 
                        WHERE access_count > 0
                        ORDER BY access_count DESC 
                        LIMIT 10
                    """)
                    most_accessed = cursor.fetchall()

                    insights["most_accessed_files"] = [
                        {
                            "filename": row[0],
                            "path": row[1],
                            "access_count": row[2],
                            "last_accessed": row[3],
                        }
                        for row in most_accessed
                    ]

                # Recently modified files
                cursor.execute("""
                    SELECT filename, file_path, last_modified, file_size
                    FROM file_metadata 
                    ORDER BY last_modified DESC 
                    LIMIT 20
                """)
                recent_files = cursor.fetchall()

                insights["recently_modified"] = [
                    {
                        "filename": row[0],
                        "path": row[1],
                        "last_modified": row[2],
                        "size": row[3],
                    }
                    for row in recent_files
                ]

                # File creation patterns by hour/day
                cursor.execute("""
                    SELECT 
                        strftime('%H', created_date) as hour,
                        COUNT(*) as count
                    FROM file_metadata 
                    WHERE created_date IS NOT NULL
                    GROUP BY hour
                    ORDER BY hour
                """)
                hourly_creation = cursor.fetchall()

                insights["creation_patterns"] = {
                    "by_hour": [
                        {"hour": int(row[0]), "count": row[1]}
                        for row in hourly_creation
                    ]
                }

                return insights
        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {e}")
            return {"error": str(e)}

    def _analyze_storage_patterns(self) -> Dict[str, Any]:
        """Analyze storage usage and optimization opportunities."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Large files analysis
                cursor.execute("""
                    SELECT filename, file_path, file_size, file_extension
                    FROM file_metadata 
                    ORDER BY file_size DESC 
                    LIMIT 20
                """)
                large_files = cursor.fetchall()

                # Storage by directory
                cursor.execute("""
                    SELECT 
                        SUBSTR(file_path, 1, INSTR(file_path || '/', '/') - 1) as root_dir,
                        COUNT(*) as file_count,
                        SUM(file_size) as total_size
                    FROM file_metadata
                    GROUP BY root_dir
                    ORDER BY total_size DESC
                """)
                storage_by_dir = cursor.fetchall()

                # Duplicate file potential (same size and extension)
                cursor.execute("""
                    SELECT file_size, file_extension, COUNT(*) as count
                    FROM file_metadata
                    WHERE file_size > 1024  -- Files larger than 1KB
                    GROUP BY file_size, file_extension
                    HAVING COUNT(*) > 1
                    ORDER BY file_size * COUNT(*) DESC
                    LIMIT 10
                """)
                potential_duplicates = cursor.fetchall()

                # Empty files
                cursor.execute("""
                    SELECT COUNT(*) as empty_count
                    FROM file_metadata
                    WHERE file_size = 0
                """)
                empty_files_count = cursor.fetchone()[0]

                return {
                    "largest_files": [
                        {
                            "filename": row[0],
                            "path": row[1],
                            "size": row[2],
                            "extension": row[3],
                            "size_mb": row[2] / (1024 * 1024),
                        }
                        for row in large_files
                    ],
                    "storage_by_directory": [
                        {
                            "directory": row[0] or "Root",
                            "file_count": row[1],
                            "total_size": row[2],
                            "size_gb": row[2] / (1024 * 1024 * 1024),
                        }
                        for row in storage_by_dir[:10]
                    ],
                    "potential_duplicates": [
                        {
                            "size": row[0],
                            "extension": row[1],
                            "count": row[2],
                            "potential_savings": row[0] * (row[2] - 1),
                        }
                        for row in potential_duplicates
                    ],
                    "empty_files_count": empty_files_count,
                }
        except Exception as e:
            logger.error(f"Error analyzing storage patterns: {e}")
            return {"error": str(e)}

    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality across analyzable files."""
        try:
            data_files = self._get_analyzable_files()
            quality_issues = []
            quality_summary = {
                "total_data_files": len(data_files),
                "analyzed_files": 0,
                "files_with_issues": 0,
                "common_issues": Counter(),
            }

            for file_path in data_files[:10]:  # Analyze up to 10 files
                try:
                    if self.data_analyzer.can_analyze_file(file_path):
                        analysis = self.data_analyzer.analyze_file(file_path)
                        quality_summary["analyzed_files"] += 1

                        if "missing_data" in analysis:
                            missing_data = analysis["missing_data"]
                            if missing_data.get("total_missing", 0) > 0:
                                quality_issues.append(
                                    {
                                        "file": Path(file_path).name,
                                        "issue": "missing_data",
                                        "details": f"{missing_data['total_missing']} missing values",
                                    }
                                )
                                quality_summary["files_with_issues"] += 1
                                quality_summary["common_issues"]["missing_data"] += 1

                        if "basic_stats" in analysis:
                            stats = analysis["basic_stats"]
                            if stats.get("duplicate_rows", 0) > 0:
                                quality_issues.append(
                                    {
                                        "file": Path(file_path).name,
                                        "issue": "duplicate_rows",
                                        "details": f"{stats['duplicate_rows']} duplicate rows",
                                    }
                                )
                                quality_summary["common_issues"]["duplicate_rows"] += 1

                except Exception as e:
                    logger.debug(f"Error analyzing data quality for {file_path}: {e}")

            return {
                "summary": quality_summary,
                "issues": quality_issues[:20],  # Top 20 issues
                "recommendations": self._generate_data_quality_recommendations(
                    quality_summary
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing data quality: {e}")
            return {"error": str(e)}

    def _analyze_security_patterns(self) -> Dict[str, Any]:
        """Analyze potential security concerns in file patterns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Executable files
                cursor.execute("""
                    SELECT filename, file_path, file_size
                    FROM file_metadata 
                    WHERE file_extension IN ('.exe', '.bat', '.cmd', '.ps1', '.vbs', '.scr')
                    ORDER BY last_modified DESC
                """)
                executable_files = cursor.fetchall()

                # Large files in temp directories
                cursor.execute("""
                    SELECT filename, file_path, file_size
                    FROM file_metadata 
                    WHERE (file_path LIKE '%temp%' OR file_path LIKE '%tmp%')
                    AND file_size > 10485760  -- 10MB
                    ORDER BY file_size DESC
                """)
                large_temp_files = cursor.fetchall()

                # Files with suspicious extensions
                suspicious_extensions = [".tmp", ".log", ".bak", ".old", ".~"]
                cursor.execute(
                    f"""
                    SELECT file_extension, COUNT(*) as count, SUM(file_size) as total_size
                    FROM file_metadata 
                    WHERE file_extension IN ({",".join(["?" for _ in suspicious_extensions])})
                    GROUP BY file_extension
                """,
                    suspicious_extensions,
                )
                suspicious_files = cursor.fetchall()

                return {
                    "executable_files": {
                        "count": len(executable_files),
                        "recent": [
                            {"filename": row[0], "path": row[1], "size": row[2]}
                            for row in executable_files[:10]
                        ],
                    },
                    "large_temp_files": [
                        {
                            "filename": row[0],
                            "path": row[1],
                            "size": row[2],
                            "size_mb": row[2] / (1024 * 1024),
                        }
                        for row in large_temp_files[:10]
                    ],
                    "suspicious_file_patterns": [
                        {"extension": row[0], "count": row[1], "total_size": row[2]}
                        for row in suspicious_files
                    ],
                }
        except Exception as e:
            logger.error(f"Error analyzing security patterns: {e}")
            return {"error": str(e)}

    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze patterns that might affect system performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Fragment analysis - many small files in same directory
                cursor.execute("""
                    SELECT 
                        SUBSTR(file_path, 1, LENGTH(file_path) - LENGTH(filename) - 1) as directory,
                        COUNT(*) as file_count,
                        AVG(file_size) as avg_size
                    FROM file_metadata
                    GROUP BY directory
                    HAVING file_count > 100
                    ORDER BY file_count DESC
                    LIMIT 10
                """)
                fragmented_dirs = cursor.fetchall()

                # Very large files that might slow access
                cursor.execute("""
                    SELECT filename, file_path, file_size
                    FROM file_metadata
                    WHERE file_size > 1073741824  -- 1GB
                    ORDER BY file_size DESC
                """)
                very_large_files = cursor.fetchall()

                # Deep directory structures
                cursor.execute("""
                    SELECT file_path,
                           LENGTH(file_path) - LENGTH(REPLACE(file_path, '/', '')) as depth
                    FROM file_metadata
                    WHERE LENGTH(file_path) - LENGTH(REPLACE(file_path, '/', '')) > 8
                    ORDER BY depth DESC
                    LIMIT 10
                """)
                deep_paths = cursor.fetchall()

                return {
                    "fragmented_directories": [
                        {
                            "directory": row[0],
                            "file_count": row[1],
                            "average_size": row[2],
                        }
                        for row in fragmented_dirs
                    ],
                    "very_large_files": [
                        {
                            "filename": row[0],
                            "path": row[1],
                            "size": row[2],
                            "size_gb": row[2] / (1024 * 1024 * 1024),
                        }
                        for row in very_large_files
                    ],
                    "deep_directory_paths": [
                        {"path": row[0], "depth": row[1]} for row in deep_paths
                    ],
                }
        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e}")
            return {"error": str(e)}

    def _generate_ai_recommendations(self, query: Optional[str] = None) -> List[str]:
        """Generate AI-powered recommendations based on insights."""
        if not self.ai_assistant.is_available():
            return ["AI assistant not available for recommendations"]

        try:
            # Prepare context from recent insights
            context = {
                "file_system_summary": self._get_file_system_summary(),
                "user_query": query,
            }

            prompt = f"""Based on the file system analysis, provide 5-7 actionable recommendations for better file management and organization.

Context: {json.dumps(context, indent=2)}

Focus on:
1. Storage optimization
2. Organization improvements
3. Performance enhancements
4. Security considerations
5. Data quality improvements

Provide specific, actionable recommendations."""

            response = self.ai_assistant.generate_response(prompt, context)

            # Parse recommendations from response
            recommendations = []
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("•") or line.startswith("-") or line.startswith("*")
                ):
                    recommendations.append(line.lstrip("•-* "))
                elif line and any(line.startswith(str(i)) for i in range(1, 10)):
                    recommendations.append(line)

            return recommendations[:7] if recommendations else [response]

        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in file creation, modification, and usage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # File creation trends over time
                cursor.execute("""
                    SELECT 
                        DATE(created_date) as date,
                        COUNT(*) as files_created,
                        SUM(file_size) as total_size
                    FROM file_metadata
                    WHERE created_date >= date('now', '-30 days')
                    GROUP BY DATE(created_date)
                    ORDER BY date
                """)
                creation_trends = cursor.fetchall()

                # File type trends
                cursor.execute("""
                    SELECT 
                        file_extension,
                        DATE(created_date) as date,
                        COUNT(*) as count
                    FROM file_metadata
                    WHERE created_date >= date('now', '-30 days')
                    GROUP BY file_extension, DATE(created_date)
                    ORDER BY date, count DESC
                """)
                type_trends = cursor.fetchall()

                return {
                    "creation_trends": [
                        {"date": row[0], "files_created": row[1], "total_size": row[2]}
                        for row in creation_trends
                    ],
                    "file_type_trends": [
                        {"extension": row[0], "date": row[1], "count": row[2]}
                        for row in type_trends
                    ],
                }
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"error": str(e)}

    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalous patterns in file system."""
        try:
            anomalies = []

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Unusually large files for their type
                cursor.execute("""
                    SELECT 
                        filename, file_path, file_size, file_extension,
                        AVG(file_size) OVER (PARTITION BY file_extension) as avg_size_for_type
                    FROM file_metadata
                    WHERE file_size > 0
                """)
                files_data = cursor.fetchall()

                for row in files_data:
                    filename, path, size, ext, avg_size = row
                    if size > avg_size * 5:  # 5x larger than average for type
                        anomalies.append(
                            {
                                "type": "unusually_large_file",
                                "filename": filename,
                                "path": path,
                                "size": size,
                                "extension": ext,
                                "average_for_type": avg_size,
                                "ratio": size / avg_size if avg_size > 0 else 0,
                            }
                        )

                # Files created at unusual times (late night/early morning)
                cursor.execute("""
                    SELECT filename, file_path, created_date
                    FROM file_metadata
                    WHERE strftime('%H', created_date) BETWEEN '02' AND '05'
                    AND created_date >= date('now', '-7 days')
                """)
                unusual_time_files = cursor.fetchall()

                for row in unusual_time_files:
                    anomalies.append(
                        {
                            "type": "unusual_creation_time",
                            "filename": row[0],
                            "path": row[1],
                            "created_date": row[2],
                        }
                    )

            return {
                "anomalies": anomalies[:20],  # Top 20 anomalies
                "total_anomalies": len(anomalies),
            }
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"error": str(e)}

    def _get_analyzable_files(self) -> List[str]:
        """Get list of files that can be analyzed for data quality."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path
                    FROM file_metadata
                    WHERE file_extension IN ('.csv', '.xlsx', '.json', '.parquet')
                    AND file_size > 0
                    ORDER BY last_modified DESC
                    LIMIT 50
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting analyzable files: {e}")
            return []

    def _get_file_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the file system for AI context."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM file_metadata")
                total_files = cursor.fetchone()[0]

                cursor.execute("SELECT SUM(file_size) FROM file_metadata")
                total_size = cursor.fetchone()[0] or 0

                cursor.execute("""
                    SELECT file_extension, COUNT(*) 
                    FROM file_metadata 
                    GROUP BY file_extension 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 5
                """)
                top_types = cursor.fetchall()

                return {
                    "total_files": total_files,
                    "total_size_gb": total_size / (1024 * 1024 * 1024),
                    "top_file_types": [
                        {"extension": row[0], "count": row[1]} for row in top_types
                    ],
                }
        except Exception as e:
            logger.error(f"Error getting file system summary: {e}")
            return {}

    def _generate_data_quality_recommendations(
        self, quality_summary: Dict
    ) -> List[str]:
        """Generate recommendations based on data quality analysis."""
        recommendations = []

        if quality_summary["files_with_issues"] > 0:
            issue_rate = (
                quality_summary["files_with_issues"] / quality_summary["analyzed_files"]
            )
            if issue_rate > 0.5:
                recommendations.append(
                    "High rate of data quality issues detected - consider data validation processes"
                )

        common_issues = quality_summary["common_issues"]
        if common_issues.get("missing_data", 0) > 0:
            recommendations.append(
                "Multiple files have missing data - implement data cleaning procedures"
            )

        if common_issues.get("duplicate_rows", 0) > 0:
            recommendations.append(
                "Duplicate data detected across files - consider deduplication strategies"
            )

        return recommendations

    def _add_to_history(self, insights: Dict[str, Any]):
        """Add insights to history."""
        self.insight_history.append(
            {
                "timestamp": insights["timestamp"],
                "summary": {
                    "query": insights.get("query"),
                    "file_count": insights.get("file_system_insights", {})
                    .get("directory_depth_stats", {})
                    .get("average_depth", 0),
                    "insights_generated": len(
                        [
                            k
                            for k, v in insights.items()
                            if isinstance(v, dict) and not v.get("error")
                        ]
                    ),
                },
            }
        )

        # Keep history limited
        if len(self.insight_history) > self.max_history_items:
            self.insight_history = self.insight_history[-self.max_history_items :]

    def get_insight_history(self) -> List[Dict[str, Any]]:
        """Get insight generation history."""
        return self.insight_history.copy()

    def clear_cache(self):
        """Clear insight cache."""
        self.insight_cache.clear()
        self.last_analysis_time.clear()
        logger.info("Insight engine cache cleared")

    def get_cached_insights(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached insights if available and not expired."""
        if cache_key in self.insight_cache:
            cache_time = self.last_analysis_time.get(cache_key, 0)
            if time.time() - cache_time < self.cache_duration:
                return self.insight_cache[cache_key]
        return None

    def cache_insights(self, cache_key: str, insights: Dict[str, Any]):
        """Cache insights with timestamp."""
        self.insight_cache[cache_key] = insights
        self.last_analysis_time[cache_key] = time.time()

    def generate_targeted_insights(
        self, focus_area: str, query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate insights focused on a specific area."""
        cache_key = f"{focus_area}_{hash(query or '')}"
        cached = self.get_cached_insights(cache_key)
        if cached:
            return cached

        insights = {"timestamp": datetime.now().isoformat(), "focus_area": focus_area}

        if focus_area == "storage":
            insights["analysis"] = self._analyze_storage_patterns()
        elif focus_area == "security":
            insights["analysis"] = self._analyze_security_patterns()
        elif focus_area == "performance":
            insights["analysis"] = self._analyze_performance_patterns()
        elif focus_area == "data_quality":
            insights["analysis"] = self._analyze_data_quality()
        elif focus_area == "usage":
            insights["analysis"] = self._analyze_usage_patterns()
        else:
            return self.generate_comprehensive_insights(query)

        # Add AI recommendations for the focused area
        if self.ai_assistant.is_available():
            context = {"focus_area": focus_area, "analysis": insights["analysis"]}
            ai_prompt = f"Provide specific recommendations for {focus_area} optimization based on this analysis."
            insights["ai_recommendations"] = self.ai_assistant.generate_response(
                ai_prompt, context
            )

        self.cache_insights(cache_key, insights)
        return insights
