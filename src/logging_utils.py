"""
logging_utils.py

Centralized logging utilities for the project.
Safe for CLI, notebook, Ray, Slurm, DDP.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict


# -----------------------------------
# Core setup
# -----------------------------------

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    clear_handlers: bool = True,
) -> None:
    """
    Setup root logger with optional file + console handlers.

    Args:
        log_file: Path to log file. If None, file logging is disabled.
        level: Logging level.
        clear_handlers: Whether to clear existing handlers (recommended).
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    if clear_handlers:
        logger.handlers.clear()

    # formatter = logging.Formatter(
    #     "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    # )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # ---- File handler ----
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # ---- Console handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# -----------------------------------
# Per-run logging
# -----------------------------------

def setup_per_run_logging(
    log_dir: str,
    run_name: Optional[str] = None,
    level: int = logging.INFO,
) -> str:
    """
    Setup per-run logging with timestamped log file.

    Returns:
        log_path: Path to created log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{run_name}_" if run_name else ""
    log_path = os.path.join(log_dir, f"{prefix}{timestamp}.log")

    setup_logging(log_file=log_path, level=level)

    logging.info(f"Logging initialized. Log file: {log_path}")
    return log_path


# -----------------------------------
# Distributed / multi-process helpers
# -----------------------------------

def setup_distributed_logging(
    log_file: Optional[str],
    rank: int = 0,
    level: int = logging.INFO,
) -> None:
    """
    Setup logging for distributed environments.
    Only rank-0 writes to file; all ranks log to console.

    Args:
        log_file: Path to log file (rank 0 only).
        rank: Process rank.
        level: Logging level.
    """
    if rank == 0:
        setup_logging(log_file=log_file, level=level)
    else:
        setup_logging(log_file=None, level=level)


# -----------------------------------
# Exception handling
# -----------------------------------

def install_exception_hook() -> None:
    """
    Install a global exception hook to log uncaught exceptions.
    """

    def _handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.getLogger().exception(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _handle_exception


# -----------------------------------
# Structured experiment logging
# -----------------------------------

def log_experiment_header(metadata: Dict) -> None:
    """
    Log experiment metadata in a structured way.

    Args:
        metadata: Dict of experiment info (config, seed, model, etc.)
    """
    logger = logging.getLogger()
    logger.info("=" * 50)
    logger.info("Experiment Metadata")
    logger.info("-" * 50)
    for k, v in metadata.items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 50)


# -----------------------------------
# Logger access helper
# -----------------------------------

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a module-specific logger.

    Args:
        name: Usually __name__.
    """
    return logging.getLogger(name)

