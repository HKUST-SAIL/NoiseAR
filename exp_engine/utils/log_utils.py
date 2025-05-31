import os
from sys import stderr

from loguru import logger

__all__ = ["setup_logger", "get_human_readable_time"]


def get_human_readable_time(seconds: int) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours}h {minutes}m {seconds:.2f}s"


def setup_logger(save_dir: str, distributed_rank: int = 0, filename: str = "log.txt", mode: str = "a"):  # type: ignore[no-untyped-def]
    """setup logger for training and testing.
    Args:
        save_dir(str): loaction to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)

    # Remove all existing processors
    logger.remove()

    # Detailed format for file logging
    file_format = "{time:HH:mm:ss} {message}"

    # Simplified format for console output
    console_format = "<green>{time:HH:mm:ss}</green> {message}"

    # Add file logging processor
    _ = logger.add(
        save_file,
        format=file_format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        enqueue=True,
        colorize=False,
    )

    # Add console logging processor
    _ = logger.add(
        stderr,
        format=console_format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        colorize=True,
    )

    return logger
