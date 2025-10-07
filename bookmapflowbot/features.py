"""Feature engineering and visualisation utilities for BookmapFlowBot."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List

from .data_feed import DataFeed, OrderBookLevel


class FeatureEngineer:
    """Generates analytics derived from raw order-book data."""

    def __init__(self, data_feed: DataFeed) -> None:
        self.data_feed = data_feed

    def heatmap_figure(self, depth: int = 50) -> go.Figure:
        """Create a Bookmap-style heatmap using Plotly."""

        prices, times, matrix = self.data_feed.get_heatmap_matrix(depth=depth)
        if not matrix:
            return go.Figure()

        intensity = np.array(matrix).T
        fig = go.Figure(
            data=go.Heatmap(
                z=intensity,
                x=times,
                y=list(range(intensity.shape[0])),
                colorscale="Viridis",
                colorbar=dict(title="Liquidity"),
            )
        )
        fig.update_layout(
            title="Order-book Liquidity Heatmap",
            xaxis_title="Timestamp",
            yaxis_title="Price Level Index",
            template="plotly_dark",
        )
        trades = self._trade_markers(times)
        if trades:
            fig.add_trace(trades)
        return fig

    def _trade_markers(self, times: List[float]) -> go.Scatter | None:
        """Scatter plot of executed trades sized by volume."""

        history = list(self.data_feed.history)[-len(times) :]
        x: List[float] = []
        y: List[int] = []
        size: List[float] = []
        color: List[str] = []

        for idx, update in enumerate(history):
            for trade in update.trades:
                x.append(update.timestamp)
                y.append(idx)
                size.append(max(trade["size"], 0.1) * 10)
                color.append("#12c2e9" if trade.get("side") == "buy" else "#f64f59")

        if not x:
            return None

        return go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.7),
            name="Trades",
        )

    def demand_supply_zones(self, threshold: float) -> Dict[str, List[OrderBookLevel]]:
        return self.data_feed.demand_supply_zones(threshold)
