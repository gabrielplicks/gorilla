import logging

from typing import Any



def configure_logger(logger_name: str, log_level: int, log_format: str, log_filename: str) -> None:

    """Configures a logger to capture and publish logs from a specific module in a Python application.



    Args:

        logger_name: The name of the logger.

        log_level: The logging level.

        log_format: The format of the log message.

        log_filename: The name of the log file.

    """

    logging.basicConfig(level=log_level, format=log_format, filename=log_filename)

    logger = logging.getLogger(logger_name)

    return logger



def my_module(logger: logging.Logger) -> None:

    """Captures and publishes logs from a specific module using the `logging` module.



    Args:

        logger: The logger responsible for capturing and publishing logs.

    """

    logger.info("This is a log message from my_module")

