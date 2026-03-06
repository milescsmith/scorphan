import datetime
from pathlib import Path
from sys import stdout
from typing import TextIO

from loguru import logger

# parent_module = modules[".".join(__name__.split(".")[:-1]) or "__main__"]


def init_logger(verbose: int = 0, msg_format: str | None = None, save: bool = False) -> None:
    timezone = datetime.datetime.now(datetime.UTC).astimezone().tzinfo

    match verbose:
        case 3:
            log_level = "DEBUG"
            backtrace = True
            diagnose = True
        case 2:
            log_level = "INFO"
            backtrace = False
            diagnose = False
        case 1:
            log_level = "WARNING"
            backtrace = False
            diagnose = False
        case _:
            log_level = "ERROR"
            backtrace = False
            diagnose = False

    if msg_format is None:
        if save is True:
            msg_format = "{name}:{function}:{line} - {message}"
        else:
            msg_format = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    output_sink: Path | TextIO = (
        Path(f"{__package__}_{datetime.datetime.now(tz=timezone).strftime('%d-%m-%Y--%H-%M-%S')}.log")
        if save
        else stdout
    )

    logger.add(
        sink=output_sink,
        format=msg_format,
        level=log_level,
        backtrace=backtrace,
        diagnose=diagnose,
        filter=__package__,
    )
