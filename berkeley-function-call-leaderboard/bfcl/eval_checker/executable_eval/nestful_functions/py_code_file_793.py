from typing import Union



def parse_int(string: Union[str, int]) -> int:

    """Parses a string to an integer while ignoring commas between the digits.



    Args:

        string: The input string to be parsed.



    Returns:

        The parsed integer.

    """

    if has_commas(string):

        string = string.replace(",", "")

    return int(string)



def has_commas(string: str) -> bool:

    """Checks if a string contains commas.



    Args:

        string: The input string to be checked.



    Returns:

        True if the string contains commas, False otherwise.

    """

    return "," in string

