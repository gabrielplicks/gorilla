from datetime import datetime

from typing import Tuple



def time_remaining_in_year() -> str:

    """Calculates the amount of time in seconds remaining until the end of the current calendar year.

    Returns:

        A string formatted as `{hours}:{minutes}:{seconds}`.

    """

    current_time = datetime.now()

    seconds_in_year = 365 * 24 * 60 * 60

    seconds_remaining = seconds_in_year - current_time.timestamp()

    hours, seconds_remaining = divmod(seconds_remaining, 3600)

    minutes, seconds = divmod(seconds_remaining, 60)

    return f'{int(hours)}:{int(minutes)}:{int(seconds)}'

