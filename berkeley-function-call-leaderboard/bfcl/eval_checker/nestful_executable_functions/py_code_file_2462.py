from typing import Union



def last_three_characters_replaced_by_asterisk(string: str) -> str:

    """Replaces the last three characters of a string with `***`.



    Args:

        string: The input string.



    Returns:

        The input string with the last three characters replaced with `***`.

        If the input string is shorter than three characters, the function

        returns the original string without modification.

    """

    if len(string) < 3:

        return string

    else:

        return string[:-3] + '***'

