from typing import List



def format_number(number: int) -> str:

    """Formats a number with a prefix "num" and a suffix 0.



    Args:

        number: The number to format.



    Returns:

        The formatted string.

    """

    return f"num{number}0"



def format_numbers(numbers: List[int]) -> List[str]:

    """Formats a list of numbers with a prefix "num" and a suffix 0.



    Args:

        numbers: The list of numbers to format.



    Returns:

        The list of formatted strings.

    """

    return list(map(format_number, numbers))

