import logging
import sys

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"
COLOR_BOLD_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

SUPPRESS = logging.CRITICAL + 10
logging.getLogger().setLevel(SUPPRESS)
COLORS = {
    logging.DEBUG: MAGENTA,
    logging.INFO: BLUE,
    logging.WARNING: YELLOW,
    logging.CRITICAL: RED,
    logging.ERROR: RED,
}
LEVEL_STRING = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.CRITICAL: "CRITICAL",
    logging.ERROR: "ERROR",
    SUPPRESS: "SUPPRESS",
}
LOG_LEVEL = logging.INFO


class JsonFormatter(logging.Formatter):
    FORMAT = "[%(levelname)s][%(name)s:%(filename)s:%(funcName)s:%(lineno)d] %(message)s"  # + ""
    FORMAT_WARNING = "[%(funcName)s] %(message)s"
    FORMAT_INFO = "%(message)s"

    def __init__(self):
        logging.Formatter.__init__(self, self.FORMAT)

    def format(self, record):
        if record.levelno == logging.WARNING:
            self._style._fmt = self.FORMAT_WARNING
        elif record.levelno == logging.INFO:
            self._style._fmt = self.FORMAT_INFO
        else:
            self._style._fmt = self.FORMAT
        result = logging.Formatter.format(self, record)
        result = COLOR_SEQ % (30 + COLORS[record.levelno]) + result + RESET_SEQ
        return result


# Custom logger class with multiple destinations
class nnLogger(logging.Logger):
    levels = []
    loggers = []
    params = {"level": None}

    def __init__(self, name, level):
        logging.Logger.__init__(self, name)
        self.setLevel(max(level, LOG_LEVEL))

        # file = logging.FileHandler('example.log')
        # color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        self.console = logging.StreamHandler(sys.stdout)
        color_formatter = JsonFormatter()

        self.console.setFormatter(color_formatter)
        # self.console.setLevel(logging.CRITICAL)

        # logging.getLogger().addHandler(self.console)
        self.addHandler(self.console)
        self.loggers.append(self)
        self.levels.append(level)
        # self.addHandler(file)

    def setAllLevel(self, level):
        if self.params["level"] is None or self.params["level"] != level:
            self._log(
                logging.INFO,
                COLOR_SEQ % (30 + BLUE)
                + (f" Loggers to {LEVEL_STRING[level]} ").center(80, "=")
                + RESET_SEQ,
                None,
            )
            self.params["level"] = level
        for ind, logger in enumerate(self.loggers):
            logger.setLevel(level)

    def resetAllLevel(self):
        if self.params["level"] != 0:
            self._log(
                logging.INFO,
                COLOR_SEQ % (30 + BLUE)
                + (" Standard Level Log ").center(80, "=")
                + RESET_SEQ,
                None,
            )
            self.params["level"] = None
        for ind, logger in enumerate(self.loggers):
            logger.setLevel(self.levels[ind])
