from typing import Union



def check_valid_integer(input_str: Union[str, int]) -> Union[int, None]:

    """Checks if a given string is a valid integer. If it is, converts it into an integer and returns the result. Otherwise, returns None.



    Args:

        input_str: The input string to check.



    Returns:

        The input converted to an integer if it is a valid integer, otherwise None.

    """

    if isinstance(input_str, int):

        return input_str

    else:

        try:

            return int(input_str)

        except ValueError:

            return None

