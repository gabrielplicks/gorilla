from typing import Union



def format_percentage(value: Union[int, float]) -> str:

    """Formats a floating point value as a percentage string, with a maximum of two decimal places, while handling edge cases such as zero values.



    Args:

        value: The floating point value to be formatted as a percentage.



    Returns:

        The formatted percentage string.

    """

    percentage = value * 100

    if percentage == 0:

        return "0%"

    return f"{percentage:.2f}%"

