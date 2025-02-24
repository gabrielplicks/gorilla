from typing import Union



def replace_string_with_spaces(input_string: Union[str, None]) -> str:

    """Replaces all spaces in a string with "%20".



    Args:

        input_string: The input string.



    Raises:

        TypeError: If the input is not a string.

    """

    if isinstance(input_string, str):

        return input_string.replace(" ", "%20")

    else:

        raise TypeError("Input must be a string.")

