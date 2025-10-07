# BookmapFlowBot

BookmapFlowBot is an asyncio-based trading research framework that consumes Level 2 order-book data, engineers Bookmap-style visualisations, and executes systematic order-flow strategies with full risk management and analytics.

## Features

- **Data Layer** – Modular connector framework with a simulated feed out of the box. Order-book snapshots and trades are stored in SQLite with rolling in-memory history and metrics such as imbalance, order-flow delta, liquidity walls, VWAP, and order-flow velocity.
- **Feature Engineering & Visualisation** – Generate Plotly heatmaps and demand/supply zones to mimic the Bookmap experience.
- **Strategy Logic** – Imbalance-based strategy supporting trend-following and mean-reversion modes, liquidity wall confirmation, risk-based targets, and optional Deep LOB ML filtering.
- **Execution Layer** – Paper trading engine with position management, CSV/SQLite logging, 1% risk-based sizing, and circuit breaker enforcing a 5% max daily drawdown.
- **Dashboard** – Flask + Socket.IO dashboard (http://localhost:5000) with real-time heatmap, signal state, P&L, trade history, and control buttons.
- **Analytics** – Risk manager computes win rate, average R:R, Sharpe ratio proxy, profit factor, and equity curve.

## Project Structure

```
bookmapflowbot/
├── __init__.py
├── data_feed.py
├── execution.py
├── features.py
├── ml.py
├── risk.py
├── strategy.py
├── dashboard.py
└── utils/
    ├── __init__.py
    └── logger.py
main.py
config.yaml
.env
requirements.txt
```

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**

   - Edit `.env` with your API keys and broker selection (`simulated`, `binance`, etc.).
   - Update `config.yaml` for symbol, thresholds, execution settings, and risk parameters.

3. **Launch the services**

   ```bash
   export $(cat .env | xargs)  # or use a dotenv loader in your shell
   python main.py --config config.yaml
   ```

   The command starts the asyncio event loop, spins up the data feed, strategy, execution engine, and brings the Flask dashboard online at http://localhost:5000.

4. **Interact with the bot**

   - Open the dashboard in a browser to monitor the heatmap, signals, and P&L in real time.
   - Use the **Start**, **Pause**, and **Stop** buttons to control streaming updates and trading activity.
   - Logs and databases are written to the `data/` directory while the bot is running.

## Machine Learning Filter

A simplified Deep LOB-style model is provided in `bookmapflowbot/ml.py`. Enable it by setting `ml.enabled: true` in `config.yaml` and providing a trained model checkpoint path. The model outputs {-1, 0, +1} directional labels used to veto strategy signals if confidence exceeds the configured threshold.

## Logging & Analytics

- Order-book snapshots are persisted to `data/bookmap.db`.
- Trade fills are stored in `data/execution.db` and `data/trade_log.csv`.
- Performance metrics can be obtained programmatically via `RiskManager.performance_metrics()` and `RiskManager.equity_curve()`.

## Extending Connectors

Implement `BaseConnector` in `bookmapflowbot/data_feed.py` to integrate additional brokers (Binance, Coinbase, Tradovate, Interactive Brokers). The connector must yield `OrderBookUpdate` instances with bids, asks, and recent trades.

## Disclaimer

This project is for educational and research purposes. It ships with a simulated connector only. Connecting to live trading venues and risking real capital requires careful validation, compliance, and additional safeguards.
