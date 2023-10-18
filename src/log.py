#!/usr/bin/env python
"""Provides log related functions."""

from logging import Logger, getLogger, basicConfig

from config import LOG_FILENAME, LOG_FORMAT, LOG_LEVEL

def setup_logger(
        logger_name: str,
        filename: str = LOG_FILENAME,
        format: str = LOG_FORMAT,
        level: int = LOG_LEVEL
    ) -> Logger:
    basicConfig(filename=filename, format=format, level=level)
    return getLogger(logger_name)
