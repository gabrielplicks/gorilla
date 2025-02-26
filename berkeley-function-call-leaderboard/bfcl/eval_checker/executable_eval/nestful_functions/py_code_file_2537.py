from datetime import datetime



def is_valid_time_format(time_format: str) -> bool:

    """Checks if the given time_format is a valid time format using the datetime module in the standard library.

    The time_format is in the format of '%I:%M %p'.

    Args:

        time_format: The time format to check.

    """

    try:

        datetime.strptime("12:34 PM", time_format)

        return True

    except ValueError:

        return False

