from typing import Union

import datetime



def combine_date_time(date_string: str, time_string: str) -> str:

    """Combines a date string and a time string into a single datetime string.



    Args:

        date_string: The date string in the format `YYYY-MM-DD`.

        time_string: The time string in the format `HH:MM:SS`.



    Returns:

        The combined datetime string in the format `YYYY-MM-DD HH:MM:SS`.

    """

    date_parts = date_string.split('-')

    time_parts = time_string.split(':')

    datetime_object = datetime.datetime(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]),

                                        int(time_parts[0]), int(time_parts[1]), int(time_parts[2]))

    return datetime_object.strftime('%Y-%m-%d %H:%M:%S')

