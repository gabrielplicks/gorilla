from typing import List



def string_to_num(string: str) -> int:

    """Calculates the sum of numbers in a string separated by 'x'.



    Args:

        string: The input string.



    Returns:

        The sum of the numbers in the string.

    """

    parts: List[str] = string.split('x')

    result: int = 0

    for part in parts:

        result += int(part)

    return result

