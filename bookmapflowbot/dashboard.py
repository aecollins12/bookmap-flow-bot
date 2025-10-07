"""Flask + Socket.IO dashboard for BookmapFlowBot."""

from __future__ import annotations

import asyncio
import json
from threading import Thread
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request
from flask_socketio import SocketIO

from .execution import ExecutionEngine
from .features import FeatureEngineer
from .risk import RiskManager
from .strategy import Strategy
from .utils.logger import get_logger

LOGGER = get_logger("dashboard")


DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BookmapFlowBot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background: #111; color: #eee; }
        .container { display: flex; flex-direction: column; gap: 20px; padding: 20px; }
        .row { display: flex; gap: 20px; }
        .card { background: #1e1e1e; border-radius: 8px; padding: 16px; flex: 1; }
        button { padding: 10px 18px; margin-right: 10px; border: none; border-radius: 6px; cursor: pointer; }
        button.start { background: #28a745; color: white; }
        button.stop { background: #dc3545; color: white; }
        button.pause { background: #ffc107; color: #000; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; border-bottom: 1px solid #333; }
    </style>
</head>
<body>
<div class="container">
    <div class="row">
        <div class="card" style="flex:2;">
            <h2>Order-book Heatmap</h2>
            <div id="heatmap"></div>
        </div>
        <div class="card" style="flex:1;">
            <h2>Status</h2>
            <p>Signal: <span id="signal">neutral</span></p>
            <p>Open PnL: <span id="pnl">0</span></p>
            <p>Mode: <span id="mode"></span></p>
            <button class="start" onclick="control('start')">Start</button>
            <button class="pause" onclick="control('pause')">Pause</button>
            <button class="stop" onclick="control('stop')">Stop</button>
        </div>
    </div>
    <div class="row">
        <div class="card">
            <h2>Trade History</h2>
            <table>
                <thead>
                    <tr><th>Timestamp</th><th>Side</th><th>Size</th><th>Entry</th><th>Exit</th><th>PnL</th></tr>
                </thead>
                <tbody id="trade-history"></tbody>
            </table>
        </div>
    </div>
</div>
<script>
    const socket = io();
    socket.on('update', (data) => {
        document.getElementById('signal').innerText = data.signal;
        document.getElementById('pnl').innerText = data.pnl.toFixed(2);
        document.getElementById('mode').innerText = data.mode;
        Plotly.newPlot('heatmap', data.heatmap.data, data.heatmap.layout);
        const tbody = document.getElementById('trade-history');
        tbody.innerHTML = '';
        data.trades.forEach((trade) => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${trade.timestamp.toFixed(0)}</td><td>${trade.side}</td><td>${trade.size.toFixed(3)}</td><td>${trade.entry.toFixed(2)}</td><td>${trade.exit.toFixed(2)}</td><td>${trade.pnl.toFixed(2)}</td>`;
            tbody.appendChild(row);
        });
    });

    function control(action) {
        fetch('/control', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action})});
    }
</script>
</body>
</html>
"""


class Dashboard:
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        strategy: Strategy,
        execution: ExecutionEngine,
        risk_manager: RiskManager,
    ) -> None:
        self.feature_engineer = feature_engineer
        self.strategy = strategy
        self.execution = execution
        self.risk_manager = risk_manager
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode="threading", cors_allowed_origins="*")
        self._paused = False
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.route("/")
        def index() -> Any:
            return render_template_string(DASHBOARD_TEMPLATE)

        @self.app.route("/control", methods=["POST"])
        def control() -> Any:
            action = request.json.get("action")
            if action == "pause":
                self._paused = True
            elif action == "start":
                self._paused = False
            elif action == "stop":
                self._paused = True
            return jsonify({"status": "ok", "paused": self._paused})

    def start_background_loop(self) -> None:
        def runner() -> None:
            LOGGER.info("Starting Flask dashboard on http://localhost:5000")
            self.socketio.run(self.app, host="0.0.0.0", port=5000)

        thread = Thread(target=runner, daemon=True)
        thread.start()

    async def broadcast(self) -> None:
        while True:
            await asyncio.sleep(1)
            if self._paused:
                continue
            heatmap = self.feature_engineer.heatmap_figure().to_dict()
            signal_state = "neutral"
            if self.strategy.current_signal:
                signal_state = "bullish" if self.strategy.current_signal.direction == 1 else "bearish"
            pnl = sum(trade.pnl for trade in self.risk_manager.trade_history)
            trades = [
                {
                    "timestamp": trade.timestamp,
                    "side": trade.side,
                    "size": trade.size,
                    "entry": trade.entry_price,
                    "exit": trade.exit_price,
                    "pnl": trade.pnl,
                }
                for trade in self.risk_manager.trade_history[-20:]
            ]
            payload = {
                "signal": signal_state,
                "pnl": pnl,
                "mode": self.strategy.mode.value,
                "heatmap": heatmap,
                "trades": trades,
            }
            self.socketio.emit("update", payload)
