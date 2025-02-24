import datetime



def get_readable_time(timestamp: int) -> str:

    """Converts a Unix timestamp to a human-readable time string in the format 'HH:MM:SS'.



    Args:

        timestamp: The Unix timestamp to convert.



    Raises:

        TypeError: If the provided timestamp is not an integer.

        ValueError: If the provided timestamp is not a valid Unix timestamp.

    """

    try:

        dt = datetime.datetime.fromtimestamp(timestamp)

        time_str = dt.strftime('%H:%M:%S')

        return time_str

    except (TypeError, ValueError):

        raise

