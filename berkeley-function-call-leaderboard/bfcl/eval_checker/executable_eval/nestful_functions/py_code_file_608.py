from typing import Union

import datetime



def validate_and_correct_time(time_string: str) -> str:

    """Validates and corrects a time string in 24-hour format.



    Args:

        time_string: A string representing a time value in 24-hour format, e.g., "19:35" or "23:59".



    Raises:

        Exception: If the input is not a valid time string format.



    Returns:

        The corrected time string with leading zeroes.

    """

    try:

        time_obj = datetime.datetime.strptime(time_string, '%H:%M')

    except ValueError:

        raise Exception('Invalid time string format.')



    corrected_time = time_obj.strftime('%H:%M')

    return corrected_time

