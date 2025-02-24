from datetime import datetime



def parse_log_message(message: str) -> datetime:

    """

    Parses a log message and returns a date object.

    Args:

        message: A string of the format `YYYY-MM-DD HH:MM:SS.sss` (where `sss` are milliseconds).

    """

    log_format = "%Y-%m-%d %H:%M:%S.%f"

    try:

        date = datetime.strptime(message, log_format)

    except ValueError:

        raise ValueError(f"Invalid log message: {message}")

    else:

        return date

