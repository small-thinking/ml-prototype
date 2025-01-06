"""This file contains the utility functions for the project.
Run this script with the command:
    python -m ml_prototype.mm_embedding.util
"""
import logging
import inspect
from colorama import Fore, Style
import os


class Logger:
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    COLORS = {
        "debug": Fore.CYAN,
        "info": Fore.GREEN,
        "warn": Fore.YELLOW,
        "error": Fore.RED,
        "critical": Fore.MAGENTA,
    }

    def __init__(self, level: str = "info"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.LEVELS.get(level, logging.INFO))
        self.logger.propagate = False  # Prevent log messages from propagating to ancestor loggers

        # Check if handlers are already added to avoid duplicate logs
        if not self.logger.handlers:
            # Set up the console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log(self, level: str, message: str):
        caller = inspect.stack()[2]  # Get caller information
        extra = {
            "filename": os.path.relpath(caller.filename),
            "funcName": caller.function,
            "lineno": caller.lineno,
        }

        log_message = f"{extra['filename']}::{extra['funcName']}:L{extra['lineno']} - {message}"

        color = self.COLORS.get(level, Fore.WHITE)
        colored_message = f"{color}{log_message}{Style.RESET_ALL}"

        # Map the level to the corresponding logging method
        if level == "debug":
            self.logger.debug(colored_message)
        elif level == "info":
            self.logger.info(colored_message)
        elif level == "warn":
            self.logger.warning(colored_message)
        elif level == "error":
            self.logger.error(colored_message)
        elif level == "critical":
            self.logger.critical(colored_message)
        else:
            self.logger.info(colored_message)  # Default to info if unknown level

    def debug(self, message: str):
        self.log("debug", message)

    def info(self, message: str):
        self.log("info", message)

    def warn(self, message: str):
        self.log("warn", message)

    def error(self, message: str):
        self.log("error", message)

    def critical(self, message: str):
        self.log("critical", message)


# Example usage
if __name__ == "__main__":
    logger = Logger(level="debug")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warn("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
