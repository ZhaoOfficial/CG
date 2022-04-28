class Logger(object):
    """
    Allowing colored formatted prints.
    """

    color_prefix = {
        "red": "\x1b\x5b31m",
        "green": "\x1b\x5b32m",
        "yellow": "\x1b\x5b33m",
        "blue": "\x1b\x5b34m",
        "pink": "\x1b\x5b35m",
        "cyan": "\x1b\x5b36m"
    }
    color_suffix = "\x1b\x5b0m"

    def __init__(self, loggee: str, prefix = "", suffix = ""):
        self.prefix = prefix + "[{}]".format(loggee)
        self.suffix = suffix

    def _format_message(self, msg: str) -> str:
        return self.prefix + msg + self.suffix

    def _format_color(self, msg: str, color: str) -> str:
        return self.color_prefix[color] + msg + self.color_suffix

    def _print_to_screen(self, msg: str, color: str):
        formatted_msg = self._format_message(msg)
        formatted_msg = self._format_color(formatted_msg, color)
        print(formatted_msg)

    def warn(self, msg: str, color="yellow"):
        self._print_to_screen(msg, color)

    def error(self, msg: str, color="red"):
        self._print_to_screen(msg, color)

    def debug(self, msg: str, color="green"):
        self._print_to_screen(msg, color)

    def info(self, msg: str, color="cyan"):
        self._print_to_screen(msg, color)
