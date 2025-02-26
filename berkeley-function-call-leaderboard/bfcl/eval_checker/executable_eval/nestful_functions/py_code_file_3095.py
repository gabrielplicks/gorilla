from typing import Union



def change_case(input_string: Union[str, bytes]) -> Union[str, bytes]:

    """Converts a string to a new string based on its case.



    Args:

        input_string: The input string.



    Returns:

        The new string.

    """

    if input_string.islower():

        return input_string.upper()

    elif input_string.isupper():

        return input_string.lower()

    elif input_string.isalpha():

        return input_string.upper()

    else:

        return input_string

