"""Optional machine learning extension for BookmapFlowBot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .utils.logger import get_logger

LOGGER = get_logger("ml")


class DeepLOB(nn.Module):
    """Simplified DeepLOB-inspired architecture."""

    def __init__(self, input_channels: int = 2, conv_channels: int = 16, lstm_hidden: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels, kernel_size=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 2)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv(x)
        x = x.mean(dim=-1)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


@dataclass
class MLFilter:
    model_path: Path
    threshold: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model = DeepLOB()
        self.model.to(self.device)
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            LOGGER.info("Loaded ML model from %s", self.model_path)
        else:
            LOGGER.warning("Model path %s does not exist. Using untrained weights.", self.model_path)
        self.model.eval()

    @torch.no_grad()
    async def infer(self, lob_tensor: np.ndarray) -> Optional[int]:
        tensor = torch.from_numpy(lob_tensor).float().unsqueeze(0)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        label = int(np.argmax(probs)) - 1  # map to -1, 0, 1
        confidence = probs[np.argmax(probs)]
        LOGGER.debug("ML inference label=%s confidence=%.2f", label, confidence)
        if confidence < self.threshold:
            return None
        return label
