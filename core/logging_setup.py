# core/logging_setup.py
import logging
import logging.handlers
from pathlib import Path

_LOG_FILE = Path("hand_tracker.log")
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB
_BACKUP_COUNT = 3


def configure_logging(level: str = "WARNING") -> None:
    numeric = getattr(logging, level.upper(), logging.WARNING)
    root = logging.getLogger("hand_tracker")
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        console = logging.StreamHandler()
        console.setLevel(numeric)
        console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(console)

        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
