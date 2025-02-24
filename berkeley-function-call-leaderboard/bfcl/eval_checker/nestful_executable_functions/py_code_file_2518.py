from typing import Union



def remove_leading_trailing_characters(input_string: Union[str, int]) -> Union[str, str]:

    """Removes any leading or trailing commas, colons, or spaces from a string.

    If the input string is empty or not a string, the function returns "Invalid input".

    Args:

        input_string: The input string to be processed.

    """

    if not isinstance(input_string, str):

        return "Invalid input"

    return input_string.strip(", :")

