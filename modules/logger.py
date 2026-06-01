"""
Централизованная система логирования.
Все модули используют get_logger(__name__) для получения своего логгера.
Логи пишутся одновременно в файл logs/app.log и в консоль.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_fmt)
_file_handler.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)
_console_handler.setLevel(logging.INFO)

_root = logging.getLogger("discipline_monitor")
_root.setLevel(logging.DEBUG)
_root.addHandler(_file_handler)
_root.addHandler(_console_handler)
_root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Возвращает дочерний логгер с иерархическим именем."""
    if not name.startswith("discipline_monitor"):
        name = f"discipline_monitor.{name}"
    return logging.getLogger(name)
