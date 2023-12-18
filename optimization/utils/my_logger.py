import logging


class MyLogger(logging.Logger):
    pass


class DebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        # return record.name in ["AGNANodeEvaluation"] or record.levelno != logging.DEBUG
        return False


def init_root_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger()
    list(map(logger.removeHandler, logger.handlers))
    list(map(logger.removeFilter, logger.filters))
    config = {
        "filename": log_path,
        "filemode": "w",
        "format": "[%(asctime)s]-%(levelname)s-%(name)s %(message)s",
        "datefmt": "%H:%M:%S",
        "level": logging.DEBUG,
    }
    logging.basicConfig(**config)
    # add stream handler
    logger.addHandler(get_streamhdlr())
    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # if name not in ["AGNANodeEvaluation"]:
    #     logger.setLevel(logging.INFO)
    return logger


# def init_root_logger() -> logging.Logger:
#     """Initialize root logger."""
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#     for hdlr in logger.handlers:
#         logger.removeHandler(hdlr)
#     logger.addHandler(get_streamhdlr())
#     logger.propagate = False
#     return logger


def get_logfmt() -> logging.Formatter:
    """Get log formatter."""
    log_fmt = logging.Formatter(
        "[%(asctime)s]-%(levelname)s-%(name)s %(message)s", "%H:%M:%S"
    )
    return log_fmt


def get_streamhdlr() -> logging.StreamHandler:
    """Get stream handler."""
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(get_logfmt())
    return sh


def get_filehdlr(path: str, debug=True) -> logging.FileHandler:
    """Get file handler.
    path: file path
    """
    fh = logging.FileHandler(path, "w")
    if debug:
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO)
    fh.setFormatter(get_logfmt())
    return fh


def add_filehdlr(path: str, debug=True) -> logging.FileHandler:
    """Add file handler to root logger.
    logger_name: logger name
    path: file path
    """
    logger = logging.getLogger()
    fh = get_filehdlr(path, debug)
    logger.addHandler(fh)
    return fh


def rm_filehdlr(fh: logging.FileHandler) -> None:
    """Remove file handler from root logger.
    fh: file handler
    """
    logger = logging.getLogger()
    logger.removeHandler(fh)
