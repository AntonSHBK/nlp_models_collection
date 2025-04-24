# utils/logging.py   
import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # === Общий логгер ===
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Файл
    app_file = RotatingFileHandler(Path(log_dir) / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    app_file.setFormatter(formatter)
    root_logger.addHandler(app_file)

    # === Model логгер ===
    model_logger = logging.getLogger("model")
    model_logger.setLevel(log_level)
    model_file = RotatingFileHandler(Path(log_dir) / "model.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    model_file.setFormatter(formatter)
    model_logger.addHandler(model_file)
    model_logger.propagate = True

def get_object_logger(obj, log_dir: str = "logs", log_level: str = "INFO"):
    """
    Возвращает уникальный логгер для конкретного объекта класса.
    Лог будет писаться в отдельный файл с названием <ClassName>_<id>.log
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger_name = f"{obj.__class__.__name__}_{id(obj)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Файл для конкретного объекта
        file_handler = RotatingFileHandler(
            Path(log_dir) / f"{logger_name}.log",
            maxBytes=1*1024*1024,
            backupCount=2,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Добавить консоль
        # console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)

        logger.propagate = False

    return logger
