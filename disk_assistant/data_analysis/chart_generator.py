import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import warnings

from config import CONFIG
from utils.logger import logger

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams["figure.max_open_warning"] = 0


class ChartGenerator:
    """Generate various types of charts and visualizations for data analysis."""

    def __init__(self):
        self.chart_dir = Path(CONFIG.get("chart_output_dir", "charts"))
        self.cache_dir = Path(CONFIG.get("analysis_cache_dir", "analysis_cache"))

        # Ensure directories exist
        self.chart_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Set up matplotlib and seaborn styling
        self._setup_plotting_style()

        # Chart type mappings
        self.chart_types = {
            "line": self._create_line_chart,
            "bar": self._create_bar_chart,
            "histogram": self._create_histogram,
            "scatter": self._create_scatter_plot,
            "box": self._create_box_plot,
            "violin": self._create_violin_plot,
            "heatmap": self._create_heatmap,
            "pie": self._create_pie_chart,
            "area": self._create_area_chart,
            "correlation": self._create_correlation_matrix,
            "distribution": self._create_distribution_plot,
            "pair": self._create_pair_plot,
            "time_series": self._create_time_series_plot,
            "count": self._create_count_plot,
        }

    def _setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.style.use("default")
        sns.set_palette("husl")

        # Custom color palettes
        self.color_palettes = {
            "default": sns.color_palette("husl", 10),
            "pastel": sns.color_palette("pastel", 10),
            "dark": sns.color_palette("dark", 10),
            "colorblind": sns.color_palette("colorblind", 10),
            "viridis": sns.color_palette("viridis", 10),
            "plasma": sns.color_palette("plasma", 10),
        }

    def generate_chart(
        self,
        data: Union[pd.DataFrame, str],
        chart_type: str,
        title: str = None,
        x_col: str = None,
        y_col: str = None,
        color_col: str = None,
        size_col: str = None,
        filename: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a chart based on the specified parameters.

        Args:
            data: DataFrame or path to data file
            chart_type: Type of chart to generate
            title: Chart title
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color grouping
            size_col: Column for size variation
            filename: Output filename (auto-generated if None)
            **kwargs: Additional chart-specific parameters

        Returns:
            Dictionary with chart information
        """
        try:
            # Load data if path provided
            if isinstance(data, str):
                df = self._load_data(data)
                if df is None:
                    return {"error": "Failed to load data"}
            else:
                df = data.copy()

            # Validate chart type
            if chart_type not in self.chart_types:
                return {"error": f"Unsupported chart type: {chart_type}"}

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{chart_type}_chart_{timestamp}"

            # Create the chart
            chart_info = self.chart_types[chart_type](
                df, title, x_col, y_col, color_col, size_col, filename, **kwargs
            )

            return chart_info

        except Exception as e:
            logger.error(f"Error generating {chart_type} chart: {e}")
            return {"error": f"Chart generation failed: {str(e)}"}

    def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from various file formats."""
        try:
            path = Path(file_path)
            ext = path.suffix.lower()

            if ext == ".csv":
                return pd.read_csv(file_path)
            elif ext in [".xlsx", ".xls"]:
                return pd.read_excel(file_path)
            elif ext == ".json":
                return pd.read_json(file_path)
            elif ext == ".parquet":
                return pd.read_parquet(file_path)
            else:
                logger.error(f"Unsupported file format: {ext}")
                return None

        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None

    def _create_line_chart(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a line chart."""
        plt.figure(figsize=kwargs.get("figsize", (12, 6)))

        if color_col and color_col in df.columns:
            for category in df[color_col].unique():
                subset = df[df[color_col] == category]
                plt.plot(
                    subset[x_col],
                    subset[y_col],
                    label=category,
                    marker="o",
                    markersize=4,
                )
            plt.legend()
        else:
            plt.plot(df[x_col], df[y_col], marker="o", markersize=4, linewidth=2)

        plt.title(title or f"Line Chart: {y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_bar_chart(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a bar chart."""
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        if color_col and color_col in df.columns:
            sns.barplot(data=df, x=x_col, y=y_col, hue=color_col)
        else:
            sns.barplot(data=df, x=x_col, y=y_col)

        plt.title(title or f"Bar Chart: {y_col} by {x_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_histogram(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a histogram."""
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        bins = kwargs.get("bins", 30)
        alpha = kwargs.get("alpha", 0.7)

        if color_col and color_col in df.columns:
            for category in df[color_col].unique():
                subset = df[df[color_col] == category]
                plt.hist(
                    subset[x_col],
                    bins=bins,
                    alpha=alpha,
                    label=category,
                    density=kwargs.get("density", False),
                )
            plt.legend()
        else:
            plt.hist(
                df[x_col], bins=bins, alpha=alpha, density=kwargs.get("density", False)
            )

        plt.title(title or f"Histogram: {x_col}")
        plt.xlabel(x_col)
        plt.ylabel("Density" if kwargs.get("density", False) else "Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_scatter_plot(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a scatter plot."""
        plt.figure(figsize=kwargs.get("figsize", (10, 8)))

        scatter_kws = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 50)}

        if size_col and size_col in df.columns:
            scatter_kws["s"] = df[size_col] * kwargs.get("size_multiplier", 100)

        if color_col and color_col in df.columns:
            scatter = plt.scatter(
                df[x_col],
                df[y_col],
                c=df[color_col],
                cmap=kwargs.get("cmap", "viridis"),
                **scatter_kws,
            )
            plt.colorbar(scatter, label=color_col)
        else:
            plt.scatter(df[x_col], df[y_col], **scatter_kws)

        plt.title(title or f"Scatter Plot: {y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_box_plot(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a box plot."""
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        if x_col and y_col:
            sns.boxplot(data=df, x=x_col, y=y_col, hue=color_col)
        else:
            # Single variable box plot
            col = x_col or y_col
            sns.boxplot(data=df, y=col)

        plt.title(title or f"Box Plot: {y_col or x_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_violin_plot(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a violin plot."""
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        if x_col and y_col:
            sns.violinplot(data=df, x=x_col, y=y_col, hue=color_col)
        else:
            col = x_col or y_col
            sns.violinplot(data=df, y=col)

        plt.title(title or f"Violin Plot: {y_col or x_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_heatmap(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a heatmap."""
        plt.figure(figsize=kwargs.get("figsize", (10, 8)))

        # If specific columns provided, create pivot table
        if x_col and y_col and color_col:
            pivot_df = df.pivot_table(
                values=color_col, index=y_col, columns=x_col, aggfunc="mean"
            )
            sns.heatmap(
                pivot_df,
                annot=kwargs.get("annot", True),
                cmap=kwargs.get("cmap", "viridis"),
                fmt=".2f",
            )
        else:
            # Correlation heatmap of numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=kwargs.get("annot", True),
                cmap=kwargs.get("cmap", "coolwarm"),
                center=0,
                square=True,
            )

        plt.title(title or "Heatmap")
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_pie_chart(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a pie chart."""
        plt.figure(figsize=kwargs.get("figsize", (8, 8)))

        if y_col and y_col in df.columns:
            # Aggregate data if needed
            pie_data = (
                df.groupby(x_col)[y_col].sum() if x_col else df[y_col].value_counts()
            )
        else:
            pie_data = df[x_col].value_counts()

        colors = self.color_palettes.get(kwargs.get("palette", "default"))
        plt.pie(
            pie_data.values,
            labels=pie_data.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )

        plt.title(title or f"Pie Chart: {x_col}")
        plt.axis("equal")
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_area_chart(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create an area chart."""
        plt.figure(figsize=kwargs.get("figsize", (12, 6)))

        if color_col and color_col in df.columns:
            pivot_df = df.pivot_table(
                values=y_col, index=x_col, columns=color_col, fill_value=0
            )
            plt.stackplot(
                pivot_df.index, *pivot_df.T.values, labels=pivot_df.columns, alpha=0.7
            )
            plt.legend(loc="upper left")
        else:
            plt.fill_between(df[x_col], df[y_col], alpha=0.7)

        plt.title(title or f"Area Chart: {y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_correlation_matrix(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a correlation matrix."""
        plt.figure(figsize=kwargs.get("figsize", (10, 8)))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        mask = (
            np.triu(np.ones_like(corr_matrix, dtype=bool))
            if kwargs.get("mask_upper", False)
            else None
        )

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            mask=mask,
        )

        plt.title(title or "Correlation Matrix")
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_distribution_plot(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a distribution plot with histogram and KDE."""
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        if color_col and color_col in df.columns:
            for category in df[color_col].unique():
                subset = df[df[color_col] == category]
                sns.histplot(subset[x_col], kde=True, alpha=0.6, label=category)
            plt.legend()
        else:
            sns.histplot(df[x_col], kde=True, alpha=0.7)

        plt.title(title or f"Distribution: {x_col}")
        plt.xlabel(x_col)
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_pair_plot(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a pair plot for multiple variables."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for pair plot"}

        # Limit to first 5 numeric columns to avoid overcrowding
        cols_to_plot = numeric_cols[:5].tolist()

        if color_col and color_col in df.columns:
            g = sns.pairplot(
                df[cols_to_plot + [color_col]], hue=color_col, diag_kind="hist"
            )
        else:
            g = sns.pairplot(df[cols_to_plot], diag_kind="hist")

        g.fig.suptitle(title or "Pair Plot", y=1.02)

        return self._save_chart(filename, g.fig)

    def _create_time_series_plot(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a time series plot."""
        plt.figure(figsize=kwargs.get("figsize", (12, 6)))

        # Try to convert x_col to datetime if it's not already
        if df[x_col].dtype == "object":
            try:
                df[x_col] = pd.to_datetime(df[x_col])
            except:
                logger.warning(f"Could not convert {x_col} to datetime")

        if color_col and color_col in df.columns:
            for category in df[color_col].unique():
                subset = df[df[color_col] == category]
                plt.plot(
                    subset[x_col],
                    subset[y_col],
                    label=category,
                    marker="o",
                    markersize=3,
                )
            plt.legend()
        else:
            plt.plot(df[x_col], df[y_col], marker="o", markersize=3, linewidth=2)

        plt.title(title or f"Time Series: {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_chart(filename)

    def _create_count_plot(
        self, df, title, x_col, y_col, color_col, size_col, filename, **kwargs
    ):
        """Create a count plot."""
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        sns.countplot(data=df, x=x_col, hue=color_col, order=kwargs.get("order"))

        plt.title(title or f"Count Plot: {x_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._save_chart(filename)

    def _save_chart(self, filename: str, fig=None) -> Dict[str, Any]:
        """Save the current chart and return chart information."""
        try:
            if not filename.endswith(".png"):
                filename += ".png"

            chart_path = self.chart_dir / filename

            if fig:
                fig.savefig(chart_path, dpi=300, bbox_inches="tight")
            else:
                plt.savefig(chart_path, dpi=300, bbox_inches="tight")

            plt.close("all")  # Close all figures to free memory

            return {
                "success": True,
                "path": str(chart_path),
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "size": chart_path.stat().st_size if chart_path.exists() else 0,
            }

        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return {"error": f"Failed to save chart: {str(e)}"}

    def create_dashboard(
        self, df: pd.DataFrame, filename: str = None
    ) -> Dict[str, Any]:
        """Create a comprehensive dashboard with multiple charts."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dashboard_{timestamp}.png"

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=["object"]).columns

            # Determine subplot layout
            n_plots = 4  # Fixed dashboard with 4 key plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Data Dashboard", fontsize=16, y=0.98)

            # Plot 1: Distribution of first numeric column
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                axes[0, 0].hist(df[col].dropna(), bins=30, alpha=0.7, color="skyblue")
                axes[0, 0].set_title(f"Distribution: {col}")
                axes[0, 0].set_xlabel(col)
                axes[0, 0].set_ylabel("Frequency")
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Correlation heatmap (if multiple numeric columns)
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                im = axes[0, 1].imshow(corr_matrix, cmap="coolwarm", aspect="auto")
                axes[0, 1].set_xticks(range(len(corr_matrix.columns)))
                axes[0, 1].set_yticks(range(len(corr_matrix.columns)))
                axes[0, 1].set_xticklabels(corr_matrix.columns, rotation=45)
                axes[0, 1].set_yticklabels(corr_matrix.columns)
                axes[0, 1].set_title("Correlation Matrix")
                plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # Plot 3: Count plot for first categorical column
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                value_counts = df[col].value_counts().head(10)  # Top 10 categories
                axes[1, 0].bar(
                    range(len(value_counts)), value_counts.values, color="lightgreen"
                )
                axes[1, 0].set_xticks(range(len(value_counts)))
                axes[1, 0].set_xticklabels(value_counts.index, rotation=45)
                axes[1, 0].set_title(f"Top Categories: {col}")
                axes[1, 0].set_ylabel("Count")

            # Plot 4: Scatter plot (if we have at least 2 numeric columns)
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                axes[1, 1].scatter(df[x_col], df[y_col], alpha=0.6, color="coral")
                axes[1, 1].set_xlabel(x_col)
                axes[1, 1].set_ylabel(y_col)
                axes[1, 1].set_title(f"Scatter: {y_col} vs {x_col}")
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save_chart(filename, fig)

        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return {"error": f"Dashboard creation failed: {str(e)}"}

    def list_available_charts(self) -> List[str]:
        """Return list of available chart types."""
        return list(self.chart_types.keys())

    def get_chart_requirements(self, chart_type: str) -> Dict[str, Any]:
        """Get requirements for a specific chart type."""
        requirements = {
            "line": {"required": ["x_col", "y_col"], "optional": ["color_col"]},
            "bar": {"required": ["x_col", "y_col"], "optional": ["color_col"]},
            "histogram": {"required": ["x_col"], "optional": ["color_col", "bins"]},
            "scatter": {
                "required": ["x_col", "y_col"],
                "optional": ["color_col", "size_col"],
            },
            "box": {"required": ["x_col OR y_col"], "optional": ["color_col"]},
            "violin": {"required": ["x_col OR y_col"], "optional": ["color_col"]},
            "heatmap": {"required": [], "optional": ["x_col", "y_col", "color_col"]},
            "pie": {"required": ["x_col"], "optional": ["y_col"]},
            "area": {"required": ["x_col", "y_col"], "optional": ["color_col"]},
            "correlation": {"required": [], "optional": []},
            "distribution": {"required": ["x_col"], "optional": ["color_col"]},
            "pair": {"required": [], "optional": ["color_col"]},
            "time_series": {"required": ["x_col", "y_col"], "optional": ["color_col"]},
            "count": {"required": ["x_col"], "optional": ["color_col"]},
        }

        return requirements.get(chart_type, {"error": "Unknown chart type"})

    def clear_charts(self):
        """Clear all generated charts."""
        try:
            for chart_file in self.chart_dir.glob("*.png"):
                chart_file.unlink()
            logger.info("All charts cleared")
        except Exception as e:
            logger.error(f"Error clearing charts: {e}")
