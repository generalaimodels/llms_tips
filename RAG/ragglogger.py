import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

class CustomLogger:
    """A custom logger class with advanced features and error handling."""

    def __init__(
        self,
        name: str,
        log_file: str = "app.log",
        level: Union[int, str] = logging.INFO,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ) -> None:
        """
        Initialize the CustomLogger.

        Args:
            name (str): The name of the logger.
            log_file (str, optional): The path to the log file. Defaults to "app.log".
            level (Union[int, str], optional): The logging level. Defaults to logging.INFO.
            max_file_size (int, optional): Maximum size of each log file in bytes. Defaults to 10 MB.
            backup_count (int, optional): Number of backup log files to keep. Defaults to 5.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up file and console handlers for the logger."""
        try:
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        except Exception as e:
            print(f"Error setting up logger handlers: {str(e)}")

    def get_logger(self) -> logging.Logger:
        """Return the configured logger."""
        return self.logger

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Union[int, str] = logging.INFO
) -> logging.Logger:
    """
    Set up and return a custom logger.

    Args:
        name (str): The name of the logger.
        log_file (Optional[str], optional): The path to the log file. If None, a default path is used.
        level (Union[int, str], optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger.

    Raises:
        ValueError: If an invalid logging level is provided.
    """
    try:
        if log_file is None:
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")

        custom_logger = CustomLogger(name, log_file, level)
        return custom_logger.get_logger()
    except ValueError as ve:
        print(f"Invalid logging level: {str(ve)}")
        raise
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        raise

# # Usage example
# if __name__ == "__main__":
#     try:
#         logger = setup_logger("my_app", level=logging.DEBUG)
#         logger.debug("This is a debug message")
#         logger.info("This is an info message")
#         logger.warning("This is a warning message")
#         logger.error("This is an error message")
#         logger.critical("This is a critical message")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")