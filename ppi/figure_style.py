"""Shared figure style. Sizes follow science_skills/image_creation.md."""
from __future__ import annotations

import matplotlib.pyplot as plt

MM = 1 / 25.4

# Okabe-Ito. The same category keeps the same color across figures.
POSITIVE = "#0072B2"
NEGATIVE = "#D55E00"
NEUTRAL = "#009E73"
WARN = "#E69F00"
INK = "#000000"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.08, 1.06, letter,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="bottom", ha="right",
    )
