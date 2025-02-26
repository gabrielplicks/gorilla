from typing import Union



def extract_second_number(string: str) -> Union[int, float]:

    """Extracts the second number from a string that contains two numbers separated by a space.

    If the pattern does not match, uses a fallback value of 0.

    Args:

        string: The input string.

    """

    parts = string.split()

    if len(parts) > 1:

        try:

            return float(parts[1])

        except ValueError:

            pass

    return 0

