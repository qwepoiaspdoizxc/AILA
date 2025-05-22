import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from config import CONFIG
from utils.logger import logger


class DataAnalyzer:
    """Analyze data files and generate insights using pandas, matplotlib, and seaborn."""

    def __init__(self):
        self.cache_dir = Path(CONFIG.get("analysis_cache_dir", "analysis_cache"))
        self.chart_dir = Path(CONFIG.get("chart_output_dir", "charts"))
        self.max_file_size = CONFIG.get("max_analysis_file_size", 100 * 1024 * 1024)

        # Ensure directories exist
        self.cache_dir.mkdir(exist_ok=True)
        self.chart_dir.mkdir(exist_ok=True)

        # Set up matplotlib and seaborn defaults
        plt.style.use("default")
        sns.set_palette("husl")

    def can_analyze_file(self, file_path: str) -> bool:
        """Check if a file can be analyzed."""
        path = Path(file_path)

        # Check if file exists and size
        if not path.exists() or path.stat().st_size > self.max_file_size:
            return False

        # Check supported formats
        supported_formats = CONFIG.get(
            "supported_data_formats", [".csv", ".xlsx", ".json"]
        )
        return path.suffix.lower() in supported_formats

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a data file and return comprehensive insights."""
        if not self.can_analyze_file(file_path):
            return {"error": "File cannot be analyzed"}

        try:
            path = Path(file_path)
            file_ext = path.suffix.lower()

            # Load data based on file type
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            elif file_ext == ".xlsx":
                df = pd.read_excel(file_path)
            elif file_ext == ".json":
                df = pd.read_json(file_path)
            else:
                return {"error": f"Unsupported file format: {file_ext}"}

            # Generate comprehensive analysis
            analysis = self._generate_comprehensive_analysis(df, path.stem)

            # Cache the results
            self._cache_analysis(file_path, analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _generate_comprehensive_analysis(
        self, df: pd.DataFrame, filename: str
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis of a DataFrame."""
        analysis = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "basic_stats": self._get_basic_stats(df),
            "column_analysis": self._analyze_columns(df),
            "missing_data": self._analyze_missing_data(df),
            "correlations": self._analyze_correlations(df),
            "charts": self._generate_charts(df, filename),
            "insights": self._generate_insights(df),
            "recommendations": self._generate_recommendations(df),
        }

        return analysis

    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the DataFrame."""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
        }

    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze each column in detail."""
        column_analysis = {}

        for col in df.columns:
            analysis = {
                "type": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "unique_values": df[col].nunique(),
                "most_frequent": None,
                "statistics": {},
            }

            # For numeric columns
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                stats = df[col].describe()
                analysis["statistics"] = {
                    "mean": stats["mean"],
                    "median": stats["50%"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "q1": stats["25%"],
                    "q3": stats["75%"],
                }

            # For categorical/object columns
            elif df[col].dtype == "object":
                value_counts = df[col].value_counts()
                if not value_counts.empty:
                    analysis["most_frequent"] = {
                        "value": value_counts.index[0],
                        "count": value_counts.iloc[0],
                        "percentage": (value_counts.iloc[0] / len(df)) * 100,
                    }

            column_analysis[col] = analysis

        return column_analysis

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100

        return {
            "total_missing": missing_counts.sum(),
            "missing_by_column": missing_counts.to_dict(),
            "missing_percentage": missing_percentage.to_dict(),
            "columns_with_missing": missing_counts[missing_counts > 0].index.tolist(),
            "complete_rows": len(df) - df.isnull().any(axis=1).sum(),
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}

        corr_matrix = df[numeric_cols].corr()

        # Find strong correlations (>0.7 or <-0.7)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append(
                        {
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "numeric_columns": numeric_cols.tolist(),
        }

    def _generate_charts(self, df: pd.DataFrame, filename: str) -> List[Dict[str, str]]:
        """Generate various charts and save them."""
        charts = []

        try:
            # 1. Data overview chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Data Overview: {filename}", fontsize=16)

            # Missing data heatmap
            if df.isnull().sum().sum() > 0:
                sns.heatmap(df.isnull(), ax=axes[0, 0], cbar=True, cmap="viridis")
                axes[0, 0].set_title("Missing Data Pattern")
            else:
                axes[0, 0].text(0.5, 0.5, "No Missing Data", ha="center", va="center")
                axes[0, 0].set_title("Missing Data Pattern")

            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            axes[0, 1].pie(
                dtype_counts.values, labels=dtype_counts.index, autopct="%1.1f%%"
            )
            axes[0, 1].set_title("Data Types Distribution")

            # Numeric columns distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols].hist(ax=axes[1, 0], bins=20)
                axes[1, 0].set_title("Numeric Columns Distribution")
            else:
                axes[1, 0].text(
                    0.5, 0.5, "No Numeric Columns", ha="center", va="center"
                )
                axes[1, 0].set_title("Numeric Columns Distribution")

            # Null values by column
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                null_counts[null_counts > 0].plot(kind="bar", ax=axes[1, 1])
                axes[1, 1].set_title("Null Values by Column")
                axes[1, 1].tick_params(axis="x", rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, "No Null Values", ha="center", va="center")
                axes[1, 1].set_title("Null Values by Column")

            plt.tight_layout()
            overview_path = self.chart_dir / f"{filename}_overview.png"
            plt.savefig(overview_path, dpi=300, bbox_inches="tight")
            plt.close()

            charts.append(
                {
                    "title": "Data Overview",
                    "path": str(overview_path),
                    "type": "overview",
                }
            )

            # 2. Correlation heatmap (if numeric columns exist)
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    linewidths=0.5,
                )
                plt.title(f"Correlation Matrix: {filename}")
                plt.tight_layout()
                corr_path = self.chart_dir / f"{filename}_correlation.png"
                plt.savefig(corr_path, dpi=300, bbox_inches="tight")
                plt.close()

                charts.append(
                    {
                        "title": "Correlation Matrix",
                        "path": str(corr_path),
                        "type": "correlation",
                    }
                )

            # 3. Distribution plots for numeric columns
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for i, col in enumerate(numeric_cols):
                    if i < len(axes):
                        df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f"Distribution of {col}")
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel("Frequency")

                # Hide unused subplots
                for i in range(len(numeric_cols), len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                dist_path = self.chart_dir / f"{filename}_distributions.png"
                plt.savefig(dist_path, dpi=300, bbox_inches="tight")
                plt.close()

                charts.append(
                    {
                        "title": "Value Distributions",
                        "path": str(dist_path),
                        "type": "distribution",
                    }
                )

        except Exception as e:
            logger.error(f"Error generating charts: {e}")

        return charts

    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate data insights."""
        insights = []

        # Basic insights
        insights.append(
            f"Dataset contains {len(df)} rows and {len(df.columns)} columns"
        )

        # Missing data insights
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_pct = (missing_total / (len(df) * len(df.columns))) * 100
            insights.append(
                f"Dataset has {missing_total} missing values ({missing_pct:.1f}% of total)"
            )
        else:
            insights.append("Dataset has no missing values")

        # Duplicate insights
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            insights.append(
                f"Found {duplicates} duplicate rows ({(duplicates / len(df) * 100):.1f}%)"
            )

        # Numeric column insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Dataset has {len(numeric_cols)} numeric columns")

            # Find columns with high variance
            for col in numeric_cols:
                cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                if cv > 1:
                    insights.append(
                        f"Column '{col}' has high variability (CV = {cv:.2f})"
                    )

        # Categorical insights
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            insights.append(f"Dataset has {len(categorical_cols)} categorical columns")

            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count == len(df):
                    insights.append(f"Column '{col}' appears to be a unique identifier")
                elif unique_count < 10:
                    insights.append(
                        f"Column '{col}' has {unique_count} unique categories"
                    )

        return insights

    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate data quality and analysis recommendations."""
        recommendations = []

        # Missing data recommendations
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append(
                "Consider handling missing data through imputation or removal"
            )
            for col in missing_cols:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 50:
                    recommendations.append(
                        f"Column '{col}' has {missing_pct:.1f}% missing - consider dropping"
                    )

        # Duplicate recommendations
        if df.duplicated().sum() > 0:
            recommendations.append("Remove duplicate rows to improve data quality")

        # Data type recommendations
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if it should be datetime
                try:
                    pd.to_datetime(df[col].dropna().head(100))
                    recommendations.append(
                        f"Column '{col}' might be convertible to datetime"
                    )
                except:
                    pass

                # Check if it should be categorical
                if df[col].nunique() < len(df) * 0.5:
                    recommendations.append(
                        f"Column '{col}' might benefit from categorical conversion"
                    )

        # Analysis recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            recommendations.append(
                "Consider correlation analysis for feature selection"
            )

        if len(df) > 10000:
            recommendations.append(
                "Consider sampling for faster analysis on large datasets"
            )

        return recommendations

    def _cache_analysis(self, file_path: str, analysis: Dict[str, Any]):
        """Cache analysis results."""
        try:
            cache_key = f"{Path(file_path).stem}_{hash(file_path)}"
            cache_file = self.cache_dir / f"{cache_key}.json"

            # Make analysis JSON serializable
            serializable_analysis = self._make_json_serializable(analysis)

            with open(cache_file, "w") as f:
                json.dump(serializable_analysis, f, indent=2)

        except Exception as e:
            logger.error(f"Error caching analysis: {e}")

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def get_cached_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis if available."""
        try:
            cache_key = f"{Path(file_path).stem}_{hash(file_path)}"
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cached analysis: {e}")

        return None

    def clear_cache(self):
        """Clear analysis cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Analysis cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
