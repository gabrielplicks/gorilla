from typing import Union



def replace_AE_without_regex(text: Union[str, bytes]) -> Union[str, bytes]:

    """Replaces all instances of "A" in a given string with "4" and all instances of "E" with "3" without using regular expression.



    Args:

        text: The input string.



    Returns:

        The modified string.

    """

    new_text = ""

    for char in text:

        if char == "A":

            new_text += "4"

        elif char == "E":

            new_text += "3"

        else:

            new_text += char

    return new_text

