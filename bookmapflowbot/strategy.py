"""Trading strategy logic for BookmapFlowBot."""

from __future__ import annotations

import asyncio
import enum
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .data_feed import DataFeed, OrderBookUpdate
from .risk import RiskManager
from .ml import MLFilter
from .utils.logger import get_logger

LOGGER = get_logger("strategy")


class TrendMode(enum.Enum):
    TREND_FOLLOWING = "trend"
    MEAN_REVERSION = "mean"


@dataclass
class Signal:
    direction: int  # 1 long, -1 short
    strength: float
    reason: str
    stop_level: float
    target_level: float
    expires_at: float


@dataclass
class Position:
    side: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float


class Strategy:
    def __init__(
        self,
        data_feed: DataFeed,
        risk_manager: RiskManager,
        config: Dict[str, Any],
        ml_filter: MLFilter | None = None,
    ) -> None:
        self.data_feed = data_feed
        self.risk_manager = risk_manager
        self.config = config
        self.mode = TrendMode(config.get("mode", "trend"))
        self.current_signal: Optional[Signal] = None
        self.current_position: Optional[Position] = None
        self.listeners: list["asyncio.Queue[Signal | None]"] = []
        self.ml_filter = ml_filter

    def subscribe(self) -> "asyncio.Queue[Signal | None]":
        queue: "asyncio.Queue[Signal | None]" = asyncio.Queue(maxsize=1)
        self.listeners.append(queue)
        return queue

    async def run(self) -> None:
        while True:
            update = await self.data_feed.trade_queue.get()
            signal = await self.evaluate(update)
            await self._notify(signal)

    async def _notify(self, signal: Optional[Signal]) -> None:
        for queue in list(self.listeners):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(signal)

    async def evaluate(self, update: OrderBookUpdate) -> Optional[Signal]:
        metrics = self.data_feed.metrics
        imbalance_threshold = self.config.get("imbalance_threshold", 0.3)
        delta_threshold = self.config.get("delta_threshold", 1.0)
        expiry = self.config.get("signal_expiry", 5.0)
        liquidity_threshold = self.config.get("liquidity_wall_threshold", 10.0)

        imbalance = metrics.get("imbalance", 0.0)
        delta = metrics.get("delta", 0.0)
        liquidity = metrics.get("liquidity_walls", {})

        direction = 0
        reason = ""
        if imbalance > imbalance_threshold:
            if self.mode is TrendMode.TREND_FOLLOWING and delta > delta_threshold:
                direction = 1
                reason = "Strong bid imbalance with positive delta"
            elif self.mode is TrendMode.MEAN_REVERSION and liquidity.get("asks"):
                direction = -1
                reason = "Imbalance but ask wall rejection"
        elif imbalance < -imbalance_threshold:
            if self.mode is TrendMode.TREND_FOLLOWING and delta < -delta_threshold:
                direction = -1
                reason = "Strong ask imbalance with negative delta"
            elif self.mode is TrendMode.MEAN_REVERSION and liquidity.get("bids"):
                direction = 1
                reason = "Imbalance but bid wall absorption"

        if direction == 0:
            self.current_signal = None
            return None

        mid_price = metrics.get("mid_price", update.mid_price)
        stop_distance = self._stop_distance(direction, liquidity, mid_price, liquidity_threshold)
        if stop_distance <= 0:
            return None

        stop_level = mid_price - stop_distance if direction == 1 else mid_price + stop_distance
        target_level = mid_price + 2 * stop_distance if direction == 1 else mid_price - 2 * stop_distance

        if self.ml_filter is not None:
            try:
                label = await self.ml_filter.infer(self._build_lob_tensor())
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("ML inference failed: %s", exc)
                label = None
            if label is not None and label != direction:
                LOGGER.info("ML filter vetoed signal. label=%s direction=%s", label, direction)
                return None

        signal = Signal(
            direction=direction,
            strength=abs(imbalance),
            reason=reason,
            stop_level=stop_level,
            target_level=target_level,
            expires_at=update.timestamp + expiry,
        )
        self.current_signal = signal
        LOGGER.info("Generated signal: %s", signal)
        return signal

    def _build_lob_tensor(self) -> Any:
        history = list(self.data_feed.history)[-50:]
        if not history:
            import numpy as np

            return np.zeros((2, 10, 10))
        bids = [[level.size for level in update.bids[:10]] for update in history[-10:]]
        asks = [[level.size for level in update.asks[:10]] for update in history[-10:]]
        import numpy as np

        tensor = np.stack([np.array(bids).T, np.array(asks).T])
        return tensor

    def _stop_distance(
        self,
        direction: int,
        liquidity: Dict[str, Any],
        mid_price: float,
        liquidity_threshold: float,
    ) -> float:
        if math.isnan(mid_price):
            return 0.0
        wall_key = "bids" if direction == 1 else "asks"
        walls = liquidity.get(wall_key, [])
        if not walls:
            return self.config.get("default_stop_distance", 1.0)
        closest_wall = min(walls, key=lambda lvl: abs(lvl.price - mid_price))
        distance = abs(mid_price - closest_wall.price)
        if distance <= 0:
            distance = self.config.get("default_stop_distance", 1.0)
        return distance

    def cancel_expired_signal(self, current_time: float) -> Optional[Signal]:
        if self.current_signal and current_time > self.current_signal.expires_at:
            LOGGER.info("Signal expired: %s", self.current_signal)
            self.current_signal = None
        return self.current_signal
