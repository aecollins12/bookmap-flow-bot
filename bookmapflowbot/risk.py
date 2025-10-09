"""Risk management helpers for BookmapFlowBot."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List

from .utils.logger import get_logger

LOGGER = get_logger("risk")


@dataclass
class TradeRecord:
    timestamp: float
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    risk: float


@dataclass
class RiskManager:
    equity: float
    max_risk_per_trade: float
    max_drawdown: float
    trade_history: List[TradeRecord] = field(default_factory=list)
    initial_equity: float = field(init=False)

    def __post_init__(self) -> None:
        self.initial_equity = self.equity

    @property
    def max_daily_loss(self) -> float:
        return self.equity * self.max_drawdown

    def position_size(self, stop_distance: float) -> float:
        """Compute position size such that risk <= 1% of equity."""

        risk_capital = self.equity * self.max_risk_per_trade
        if stop_distance <= 0:
            return 0.0
        size = risk_capital / stop_distance
        LOGGER.debug("Calculated position size %.4f for stop distance %.4f", size, stop_distance)
        return max(size, 0.0)

    def record_trade(self, trade: TradeRecord) -> None:
        self.trade_history.append(trade)
        self.equity += trade.pnl

    def circuit_breaker_triggered(self) -> bool:
        loss = self.initial_equity - self.equity
        return loss >= self.initial_equity * self.max_drawdown

    def performance_metrics(self) -> Dict[str, float]:
        if not self.trade_history:
            return {"win_rate": 0.0, "avg_rr": 0.0, "sharpe": 0.0, "profit_factor": 0.0}

        wins = [trade for trade in self.trade_history if trade.pnl > 0]
        losses = [trade for trade in self.trade_history if trade.pnl <= 0]
        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0.0

        rr_ratios = [abs(trade.pnl / trade.risk) if trade.risk else 0.0 for trade in self.trade_history]
        avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0.0

        returns = [trade.pnl / max(trade.risk, 1e-9) for trade in self.trade_history]
        sharpe = (statistics.mean(returns) / (statistics.stdev(returns) or math.inf)) if len(returns) > 1 else 0.0

        gross_profit = sum(trade.pnl for trade in wins)
        gross_loss = abs(sum(trade.pnl for trade in losses))
        profit_factor = gross_profit / gross_loss if gross_loss else math.inf

        return {
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
        }

    def equity_curve(self) -> List[float]:
        curve = []
        equity = self.equity
        for trade in self.trade_history:
            equity += trade.pnl
            curve.append(equity)
        return curve
