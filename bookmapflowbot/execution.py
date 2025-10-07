"""Execution layer and paper trading integration."""

from __future__ import annotations

import asyncio
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiosqlite

from .risk import RiskManager, TradeRecord
from .strategy import Position, Signal
from .utils.logger import get_logger

LOGGER = get_logger("execution")


@dataclass
class Order:
    id: str
    side: str
    size: float
    price: float
    type: str
    timestamp: float


@dataclass
class Trade:
    order_id: str
    side: str
    size: float
    price: float
    timestamp: float


class PaperTradingBroker:
    """A minimal in-memory paper trading broker implementation."""

    def __init__(self, balance: float, db_path: str | Path = "data/execution.db") -> None:
        self.balance = balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.db_path = Path(db_path)

    async def initialise(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    order_id TEXT,
                    side TEXT,
                    size REAL,
                    price REAL,
                    timestamp REAL
                );
                """
            )
            await db.commit()

    async def submit_order(self, order: Order) -> Trade:
        self.orders[order.id] = order
        trade = Trade(
            order_id=order.id,
            side=order.side,
            size=order.size,
            price=order.price,
            timestamp=time.time(),
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO trades(order_id, side, size, price, timestamp) VALUES (?, ?, ?, ?, ?)",
                (trade.order_id, trade.side, trade.size, trade.price, trade.timestamp),
            )
            await db.commit()
        return trade


class ExecutionEngine:
    def __init__(
        self,
        risk_manager: RiskManager,
        broker: PaperTradingBroker,
        csv_path: str | Path = "data/trade_log.csv",
    ) -> None:
        self.risk_manager = risk_manager
        self.broker = broker
        self.csv_path = Path(csv_path)
        self.open_position: Optional[Position] = None
        self.order_id_counter = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "order_id",
                        "side",
                        "size",
                        "entry_price",
                        "exit_price",
                        "pnl",
                        "risk",
                        "reason",
                    ]
                )

    async def handle_signal(self, signal: Optional[Signal], market_price: float) -> None:
        if signal is None:
            return
        if self.risk_manager.circuit_breaker_triggered():
            LOGGER.warning("Circuit breaker active. No new trades will be taken today.")
            return
        if time.time() > signal.expires_at:
            LOGGER.info("Ignoring expired signal: %s", signal)
            return

        if self.open_position:
            LOGGER.debug("Position already open, ignoring new signal")
            return

        stop_distance = abs(market_price - signal.stop_level)
        size = self.risk_manager.position_size(stop_distance)
        if size <= 0:
            LOGGER.warning("Calculated size is zero, skipping trade")
            return

        side = "buy" if signal.direction == 1 else "sell"
        order = Order(
            id=self._next_order_id(),
            side=side,
            size=size,
            price=market_price,
            type="market",
            timestamp=time.time(),
        )

        trade = await self.broker.submit_order(order)
        self.open_position = Position(
            side=side,
            size=size,
            entry_price=trade.price,
            stop_loss=signal.stop_level,
            take_profit=signal.target_level,
        )
        LOGGER.info("Opened %s position of size %.4f at price %.2f", side, size, trade.price)

    async def manage_position(self, market_price: float) -> None:
        if not self.open_position:
            return

        position = self.open_position
        reached_tp = (
            market_price >= position.take_profit if position.side == "buy" else market_price <= position.take_profit
        )
        reached_sl = (
            market_price <= position.stop_loss if position.side == "buy" else market_price >= position.stop_loss
        )

        if not (reached_tp or reached_sl):
            return

        pnl = (market_price - position.entry_price) * position.size
        if position.side == "sell":
            pnl = -pnl
        risk = abs(position.entry_price - position.stop_loss) * position.size
        trade_record = TradeRecord(
            timestamp=time.time(),
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=market_price,
            pnl=pnl,
            risk=risk,
        )
        self.risk_manager.record_trade(trade_record)
        self._append_csv(trade_record)
        LOGGER.info("Closed position with PnL %.2f", pnl)
        self.open_position = None

    def _append_csv(self, trade: TradeRecord) -> None:
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    trade.timestamp,
                    len(self.risk_manager.trade_history),
                    trade.side,
                    trade.size,
                    trade.entry_price,
                    trade.exit_price,
                    trade.pnl,
                    trade.risk,
                    "",
                ]
            )

    def _next_order_id(self) -> str:
        self.order_id_counter += 1
        return f"ORD-{self.order_id_counter:06d}"
