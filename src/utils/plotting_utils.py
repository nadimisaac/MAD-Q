"""Plotting utilities for publication-quality visualizations."""

from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality plots."""
    # Set style
    sns.set_style("whitegrid", {"grid.alpha": 0.3})

    # Set font properties
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,  # Display DPI
        'savefig.dpi': 300,  # Save DPI for publication quality
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': 'gray',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
    })


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Path,
    formats: Optional[List[str]] = None
) -> None:
    """
    Save figure with consistent settings.

    Args:
        fig: Matplotlib figure to save
        name: Base name for the file (without extension)
        output_dir: Directory to save the figure
        formats: List of formats to save (default: ['png'])
    """
    if formats is None:
        formats = ['png']

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = output_dir / f"{name}.{fmt}"
        fig.savefig(
            output_path,
            format=fmt,
            dpi=300 if fmt == 'png' else None,
            bbox_inches='tight',
            pad_inches=0.1
        )
        print(f"Saved plot to: {output_path}")


def add_significance_markers(
    ax: plt.Axes,
    comparisons: List[Tuple[int, int, float, str]],
    y_offset: float = 0.05
) -> None:
    """
    Add statistical significance markers to a plot.

    Args:
        ax: Matplotlib axes object
        comparisons: List of (x1, x2, p_value, text) tuples
        y_offset: Vertical offset for markers as fraction of y-range
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_start = ax.get_ylim()[1]

    for i, (x1, x2, p_value, text) in enumerate(comparisons):
        y = y_start + (y_offset * y_range * (i + 1))

        # Draw horizontal line
        ax.plot([x1, x2], [y, y], 'k-', linewidth=1)

        # Draw vertical ticks
        tick_height = y_offset * y_range * 0.2
        ax.plot([x1, x1], [y - tick_height, y], 'k-', linewidth=1)
        ax.plot([x2, x2], [y - tick_height, y], 'k-', linewidth=1)

        # Add text
        ax.text((x1 + x2) / 2, y, text, ha='center', va='bottom', fontsize=10)


def create_colorblind_palette(n_colors: int) -> List[Tuple[float, float, float]]:
    """
    Create a colorblind-friendly palette.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of RGB tuples
    """
    # Use seaborn's colorblind palette
    return sns.color_palette("colorblind", n_colors=n_colors)


def format_pvalue(p: float) -> str:
    """
    Format p-value for display.

    Args:
        p: P-value

    Returns:
        Formatted string (e.g., "p < 0.001" or "p = 0.023")
    """
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def add_table_to_figure(
    fig: plt.Figure,
    table_data: List[List[str]],
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> None:
    """
    Add a table to a figure (useful for summary statistics).

    Args:
        fig: Matplotlib figure
        table_data: List of rows, each row is a list of strings
        bbox: Bounding box as (left, bottom, width, height) in figure coords
    """
    if bbox is None:
        bbox = (0.1, -0.3, 0.8, 0.2)

    ax_table = fig.add_axes(bbox, frameon=False)
    ax_table.axis('off')

    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        edges='horizontal'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
