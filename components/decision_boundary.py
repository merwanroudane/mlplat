"""Decision-boundary rendering for fitted 2-D classifiers."""

from __future__ import annotations

import numpy as np

from utils.plotting import decision_boundary, plot


def boundary_chart(model, X: np.ndarray, y: np.ndarray, title: str = "", highlight_val: int | None = None,
                   show_proba: bool = True, height: int = 440, highlight: np.ndarray | None = None) -> None:
    """Plot decision regions. If ``highlight_val`` is given, rows from that index on are validation
    points and are outlined so train and validation can be told apart without colour."""
    if highlight is None and highlight_val is not None:
        highlight = np.zeros(len(y), bool)
        highlight[highlight_val:] = True
    fig = decision_boundary(model, X, y, title=title, show_proba=show_proba, highlight=highlight)
    if highlight_val is not None:
        fig.data[-1].name = "validation points"
    plot(fig, height=height)
