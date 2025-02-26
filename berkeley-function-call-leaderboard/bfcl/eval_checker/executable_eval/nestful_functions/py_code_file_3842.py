from typing import Union



def format_diff_minutes(diff_minutes: Union[int, float]) -> str:

    """Formats a number of minutes as a string with a 1-unit suffix.



    Args:

        diff_minutes: A positive integer or float representing the number of minutes.



    Returns:

        A string representing the formatted time difference.

    """

    if diff_minutes == 1:

        return f"{diff_minutes} minute ago"

    elif diff_minutes == 2:

        return f"{diff_minutes} minutes ago"

    else:

        return f"{diff_minutes} minutes ago"

