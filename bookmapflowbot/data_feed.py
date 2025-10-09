"""Data layer responsible for connecting to order-book feeds and persisting data."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

import aiosqlite

from .utils.logger import get_logger

LOGGER = get_logger("data_feed")


@dataclass(slots=True)
class OrderBookLevel:
    """Represents a single price level and quantity."""

    price: float
    size: float


@dataclass(slots=True)
class OrderBookUpdate:
    """Snapshot of the L2 book along with recent trade information."""

    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    trades: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return math.nan
        return (self.bids[0].price + self.asks[0].price) / 2


class BaseConnector:
    """Base class for broker specific WebSocket connectors."""

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.kwargs = kwargs

    async def connect(self) -> AsyncIterator[OrderBookUpdate]:  # pragma: no cover - interface
        raise NotImplementedError


class SimulatedConnector(BaseConnector):
    """Simulation connector producing random order-book updates.

    This is primarily used for development and automated testing so the rest of
    the system can be exercised without connecting to a live broker.
    """

    def __init__(self, symbol: str, update_interval: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self.update_interval = update_interval
        self._price = 100.0

    async def connect(self) -> AsyncIterator[OrderBookUpdate]:
        import random

        while True:
            drift = random.uniform(-0.5, 0.5)
            self._price = max(1.0, self._price + drift)

            bids = [
                OrderBookLevel(price=self._price - i * 0.25, size=random.uniform(1, 5))
                for i in range(5)
            ]
            asks = [
                OrderBookLevel(price=self._price + i * 0.25, size=random.uniform(1, 5))
                for i in range(5)
            ]
            trades = [
                {
                    "side": random.choice(["buy", "sell"]),
                    "price": self._price + random.uniform(-0.1, 0.1),
                    "size": random.uniform(0.1, 1.0),
                }
                for _ in range(random.randint(0, 3))
            ]

            yield OrderBookUpdate(
                timestamp=time.time(),
                bids=bids,
                asks=asks,
                trades=trades,
            )
            await asyncio.sleep(self.update_interval)


class DataFeed:
    """Manages incoming order-book data, metrics and persistence."""

    def __init__(
        self,
        connector: BaseConnector,
        max_history: int = 2000,
        db_path: str | Path = "data/bookmap.db",
        liquidity_wall_threshold: float = 10.0,
    ) -> None:
        self.connector = connector
        self.max_history = max_history
        self.db_path = Path(db_path)
        self.liquidity_wall_threshold = liquidity_wall_threshold

        self.history: Deque[OrderBookUpdate] = deque(maxlen=max_history)
        self.metrics: Dict[str, Any] = {}
        self.trade_queue: "asyncio.Queue[OrderBookUpdate]" = asyncio.Queue(maxsize=max_history)
        self._db_lock = asyncio.Lock()

    async def initialise(self) -> None:
        """Initialise the SQLite schema used for persistence."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS order_book (
                    timestamp REAL PRIMARY KEY,
                    bids TEXT NOT NULL,
                    asks TEXT NOT NULL,
                    trades TEXT
                );
                """
            )
            await db.commit()
        LOGGER.debug("SQLite database initialised at %s", self.db_path)

    async def _persist(self, update: OrderBookUpdate) -> None:
        payload = (
            update.timestamp,
            json.dumps([level.__dict__ for level in update.bids]),
            json.dumps([level.__dict__ for level in update.asks]),
            json.dumps(update.trades),
        )
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO order_book(timestamp, bids, asks, trades) VALUES (?, ?, ?, ?)",
                    payload,
                )
                await db.commit()

    async def _update_metrics(self, update: OrderBookUpdate) -> None:
        bid_depth = sum(level.size for level in update.bids)
        ask_depth = sum(level.size for level in update.asks)
        imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-9)

        if len(self.history) > 0:
            last = self.history[-1]
            last_bid_depth = sum(level.size for level in last.bids)
            last_ask_depth = sum(level.size for level in last.asks)
            delta = (bid_depth - last_bid_depth) - (ask_depth - last_ask_depth)
        else:
            delta = 0.0

        liquidity_walls = {
            "bids": [level for level in update.bids if level.size >= self.liquidity_wall_threshold],
            "asks": [level for level in update.asks if level.size >= self.liquidity_wall_threshold],
        }

        velocity_counter = Counter(trade["side"] for trade in update.trades)
        velocity = {
            "buy": velocity_counter.get("buy", 0) / max(self.connector.kwargs.get("update_interval", 1.0), 1e-9),
            "sell": velocity_counter.get("sell", 0) / max(self.connector.kwargs.get("update_interval", 1.0), 1e-9),
        }

        total_volume = sum(trade["price"] * trade["size"] for trade in update.trades)
        total_size = sum(trade["size"] for trade in update.trades)
        vwap = total_volume / total_size if total_size else update.mid_price

        self.metrics = {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "imbalance": imbalance,
            "delta": delta,
            "liquidity_walls": liquidity_walls,
            "velocity": velocity,
            "vwap": vwap,
            "timestamp": update.timestamp,
            "mid_price": update.mid_price,
        }

    async def stream(self) -> None:
        """Consume data from the connector and update internal state."""

        await self.initialise()
        async for update in self.connector.connect():
            self.history.append(update)
            await self._update_metrics(update)
            await self._persist(update)

            if self.trade_queue.full():
                _ = self.trade_queue.get_nowait()
            await self.trade_queue.put(update)

    def latest(self) -> Optional[OrderBookUpdate]:
        return self.history[-1] if self.history else None

    def get_heatmap_matrix(self, depth: int = 50) -> Tuple[List[float], List[float], List[List[float]]]:
        """Return data required to plot a Bookmap-style heatmap."""

        prices: List[float] = []
        times: List[float] = []
        matrix: List[List[float]] = []

        for update in list(self.history)[-depth:]:
            times.append(update.timestamp)
            liquidity = [level.size for level in update.bids[:depth // 2]] + [
                level.size for level in update.asks[:depth // 2]
            ]
            prices.append(update.mid_price)
            matrix.append(liquidity)

        return prices, times, matrix

    def demand_supply_zones(self, threshold: float) -> Dict[str, List[OrderBookLevel]]:
        """Return price levels classified as demand or supply zones."""

        if not self.history:
            return {"demand": [], "supply": []}
        latest = self.history[-1]
        demand = [level for level in latest.bids if level.size >= threshold]
        supply = [level for level in latest.asks if level.size >= threshold]
        return {"demand": demand, "supply": supply}


async def create_connector(config: Dict[str, Any]) -> BaseConnector:
    """Factory helper to construct a connector based on configuration."""

    broker = config.get("broker", "simulated").lower()
    if broker == "simulated":
        return SimulatedConnector(symbol=config.get("symbol", "BTCUSDT"), update_interval=config.get("update_interval", 0.5))
    raise ValueError(f"Broker '{broker}' is not supported in the open-source package.")
