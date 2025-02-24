from typing import Union



def month_to_string(month_num: Union[int, float]) -> str:

    """Converts a month number to its corresponding string representation.



    Args:

        month_num: The month number.



    Returns:

        A string in the format "Month 1", "Month 2", and so on.



    Raises:

        ValueError: If the month number is outside of the valid range.

    """

    if month_num < 1 or month_num > 12:

        raise ValueError("Month number must be between 1 and 12.")



    return f"Month {month_num}"

