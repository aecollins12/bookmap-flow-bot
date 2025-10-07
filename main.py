"""Entry point for the BookmapFlowBot application."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from bookmapflowbot.dashboard import Dashboard
from bookmapflowbot.data_feed import DataFeed, create_connector
from bookmapflowbot.execution import ExecutionEngine, PaperTradingBroker
from bookmapflowbot.features import FeatureEngineer
from bookmapflowbot.ml import MLFilter
from bookmapflowbot.risk import RiskManager
from bookmapflowbot.strategy import Strategy
from bookmapflowbot.utils.logger import get_logger, setup_logging

LOGGER = get_logger(__name__)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def shutdown(loop: asyncio.AbstractEventLoop, tasks: list[asyncio.Task[Any]]) -> None:
    LOGGER.info("Shutdown initiated")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def run_bot(config: Dict[str, Any]) -> None:
    load_dotenv()
    connector = await create_connector(config)
    data_feed = DataFeed(
        connector=connector,
        max_history=2000,
        liquidity_wall_threshold=config.get("liquidity_wall_threshold", 4.0),
    )

    risk_conf = config.get("risk", {})
    risk_manager = RiskManager(
        equity=risk_conf.get("starting_equity", 100000),
        max_risk_per_trade=risk_conf.get("max_risk_per_trade", 0.01),
        max_drawdown=risk_conf.get("max_drawdown", 0.05),
    )

    feature_engineer = FeatureEngineer(data_feed)

    ml_filter = None
    ml_conf = config.get("ml", {})
    if ml_conf.get("enabled", False):
        ml_filter = MLFilter(Path(ml_conf.get("model_path", "models/deeplob.pth")), ml_conf.get("threshold", 0.6))

    strategy = Strategy(data_feed=data_feed, risk_manager=risk_manager, config=config, ml_filter=ml_filter)

    exec_conf = config.get("execution", {})
    broker = PaperTradingBroker(balance=risk_manager.equity, db_path=exec_conf.get("db_path", "data/execution.db"))
    await broker.initialise()

    execution = ExecutionEngine(
        risk_manager=risk_manager,
        broker=broker,
        csv_path=exec_conf.get("csv_path", "data/trade_log.csv"),
    )

    dashboard = None
    if config.get("dashboards", {}).get("enabled", True):
        dashboard = Dashboard(feature_engineer, strategy, execution, risk_manager)
        dashboard.start_background_loop()

    tasks: list[asyncio.Task[Any]] = []

    async def signal_consumer() -> None:
        queue = strategy.subscribe()
        while True:
            signal = await queue.get()
            latest = data_feed.latest()
            if latest is None:
                continue
            await execution.handle_signal(signal, latest.mid_price)

    async def position_manager() -> None:
        while True:
            await asyncio.sleep(0.5)
            latest = data_feed.latest()
            if latest is None:
                continue
            await execution.manage_position(latest.mid_price)
            strategy.cancel_expired_signal(latest.timestamp)

    tasks.append(asyncio.create_task(data_feed.stream(), name="data_feed"))
    tasks.append(asyncio.create_task(strategy.run(), name="strategy"))
    tasks.append(asyncio.create_task(signal_consumer(), name="signal_consumer"))
    tasks.append(asyncio.create_task(position_manager(), name="position_manager"))

    if dashboard is not None:
        tasks.append(asyncio.create_task(dashboard.broadcast(), name="dashboard_broadcast"))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        LOGGER.info("Tasks cancelled")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BookmapFlowBot")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    loop = asyncio.get_event_loop()

    tasks: list[asyncio.Task[Any]] = []

    def handle_signal(signum: int, frame: Any) -> None:
        LOGGER.info("Received signal %s", signum)
        asyncio.ensure_future(shutdown(loop, tasks))

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    try:
        main_task = loop.create_task(run_bot(config))
        tasks.append(main_task)
        loop.run_until_complete(main_task)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
