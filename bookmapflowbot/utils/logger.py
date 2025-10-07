"""Utility module providing structured logging for the BookmapFlowBot project."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(level: int = logging.INFO, log_dir: str | Path = "logs") -> logging.Logger:
    """Configure and return the root logger used across the project.

    The function creates both console and rotating file handlers so that the
    application provides immediate feedback in the terminal while also
    persisting logs for later analysis. Rotating files are particularly useful
    for long-running trading bots where disk usage needs to be controlled.

    Parameters
    ----------
    level:
        Logging level, defaults to :data:`logging.INFO`.
    log_dir:
        Directory in which log files should be stored. The directory is created
        if it does not exist.

    Returns
    -------
    logging.Logger
        The configured root logger instance.
    """

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bookmapflowbot")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "bookmapflowbot.log", maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger initialised with level %s", logging.getLevelName(level))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger derived from the project root logger."""

    return logging.getLogger("bookmapflowbot" if name is None else f"bookmapflowbot.{name}")
