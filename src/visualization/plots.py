""
Visualization module for SmartRetail Analytics.
Handles creation of various plots for data exploration and model interpretation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import matplotlib.dates as mdates
from datetime import datetime

class Plotter:
    """Handles creation of various plots for data visualization."""
    
    def __init__(self, style: str = 'seaborn', 
                 context: str = 'notebook',
                 palette: str = 'viridis'):
        """Initialize the plotter with styling options.
        
        Args:
            style: Matplotlib style to use
            context: Seaborn context (paper, notebook, talk, poster)
            palette: Color palette to use for plots
        """
        self.style = style
        self.context = context
        self.palette = palette
        
        # Set style and context
        plt.style.use(style)
        sns.set_context(context)
        sns.set_palette(palette)
    
    def time_series_plot(self, 
                        df: pd.DataFrame, 
                        x: str, 
                        y: str, 
                        hue: Optional[str] = None,
                        title: str = 'Time Series Plot',
                        xlabel: str = 'Date',
                        ylabel: str = 'Value',
                        figsize: Tuple[int, int] = (12, 6),
                        **kwargs) -> plt.Figure:
        """Create a time series plot.
        
        Args:
            df: DataFrame containing the data
            x: Column name for the x-axis (should be datetime)
            y: Column name for the y-axis
            hue: Column name for grouping
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to sns.lineplot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the line plot
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        
        # Format the x-axis for dates
        if pd.api.types.is_datetime64_any_dtype(df[x]):
            # Use AutoDateLocator for automatic date formatting
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
        
        # Set title and labels
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Improve layout
        plt.tight_layout()
        
        return fig
    
    def feature_importance_plot(self, 
                              feature_importance: Dict[str, float],
                              title: str = 'Feature Importance',
                              figsize: Tuple[int, int] = (10, 8),
                              top_n: Optional[int] = 20) -> plt.Figure:
        """Create a bar plot of feature importances.
        
        Args:
            feature_importance: Dictionary of feature names and their importance scores
            title: Plot title
            figsize: Figure size (width, height)
            top_n: Number of top features to show
            
        Returns:
            Matplotlib Figure object
        """
        # Convert to DataFrame and sort
        df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        # Limit to top N features if specified
        if top_n is not None:
            df = df.head(top_n)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        sns.barplot(data=df, x='importance', y='feature', ax=ax, palette=self.palette)
        
        # Add value labels on the bars
        for i, v in enumerate(df['importance']):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # Set title and labels
        ax.set_title(title, pad=20)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        
        # Improve layout
        plt.tight_layout()
        
        return fig
    
    def correlation_heatmap(self, 
                           df: pd.DataFrame, 
                           method: str = 'pearson',
                           title: str = 'Correlation Heatmap',
                           figsize: Tuple[int, int] = (12, 10),
                           annot: bool = True,
                           **kwargs) -> plt.Figure:
        """Create a correlation heatmap for numeric columns.
        
        Args:
            df: DataFrame containing the data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            title: Plot title
            figsize: Figure size (width, height)
            annot: Whether to annotate the heatmap with correlation values
            **kwargs: Additional arguments to pass to sns.heatmap
            
        Returns:
            Matplotlib Figure object
        """
        # Calculate correlation matrix
        corr = df.select_dtypes(include=[np.number]).corr(method=method)
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, 
                   mask=mask, 
                   cmap=cmap, 
                   vmax=.3, 
                   center=0,
                   square=True, 
                   linewidths=.5, 
                   cbar_kws={"shrink": .5},
                   annot=annot,
                   fmt=".2f",
                   ax=ax,
                   **kwargs)
        
        # Set title
        ax.set_title(title, pad=20)
        
        # Improve layout
        plt.tight_layout()
        
        return fig
    
    def distribution_plot(self, 
                         data: pd.Series,
                         title: str = 'Distribution Plot',
                         xlabel: str = 'Value',
                         ylabel: str = 'Density',
                         figsize: Tuple[int, int] = (10, 6),
                         **kwargs) -> plt.Figure:
        """Create a distribution plot (histogram with KDE).
        
        Args:
            data: Input data as a pandas Series
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to sns.histplot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the distribution plot
        sns.histplot(data, kde=True, ax=ax, **kwargs)
        
        # Set title and labels
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='g', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Add legend
        ax.legend()
        
        # Improve layout
        plt.tight_layout()
        
        return fig
    
    def scatter_plot(self,
                    df: pd.DataFrame,
                    x: str,
                    y: str,
                    hue: Optional[str] = None,
                    size: Optional[str] = None,
                    title: str = 'Scatter Plot',
                    figsize: Tuple[int, int] = (10, 8),
                    **kwargs) -> plt.Figure:
        """Create a scatter plot.
        
        Args:
            df: DataFrame containing the data
            x: Column name for the x-axis
            y: Column name for the y-axis
            hue: Column name for color encoding
            size: Column name for point sizes
            title: Plot title
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to sns.scatterplot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the scatter plot
        sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, ax=ax, **kwargs)
        
        # Set title and labels
        ax.set_title(title, pad=20)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve layout
        plt.tight_layout()
        
        return fig
    
    def box_plot(self,
                df: pd.DataFrame,
                x: str,
                y: str,
                hue: Optional[str] = None,
                title: str = 'Box Plot',
                figsize: Tuple[int, int] = (10, 6),
                **kwargs) -> plt.Figure:
        """Create a box plot.
        
        Args:
            df: DataFrame containing the data
            x: Column name for the x-axis (categorical)
            y: Column name for the y-axis (numeric)
            hue: Column name for grouping
            title: Plot title
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to sns.boxplot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the box plot
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)
        
        # Set title and labels
        ax.set_title(title, pad=20)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        
        # Rotate x-axis labels if needed
        if len(df[x].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Improve layout
        plt.tight_layout()
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300, 
                  transparent: bool = False, **kwargs) -> None:
        """Save a plot to a file.
        
        Args:
            fig: Matplotlib Figure object to save
            filename: Output filename (with extension)
            dpi: Resolution in dots per inch
            transparent: Whether the background should be transparent
            **kwargs: Additional arguments to pass to fig.savefig
        """
        # Create directory if it doesn't exist
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the figure
        fig.savefig(
            filename, 
            dpi=dpi, 
            bbox_inches='tight', 
            transparent=transparent,
            **kwargs
        )
        
        # Close the figure to free memory
        plt.close(fig)
