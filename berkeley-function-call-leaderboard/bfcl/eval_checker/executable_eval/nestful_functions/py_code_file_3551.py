import re

from typing import Union



def string_to_variable_name(string: str) -> str:

    """Converts a string into a valid variable name in Python.



    Args:

        string: The input string to convert into a variable name.



    Returns:

        A valid variable name in Python.

    """

    string = re.sub(r'[^\w]', ' ', string)

    words = string.split()

    words = [word.capitalize() for word in words]

    variable_name = ''.join(words)

    if variable_name[0].isdigit():

        variable_name = '_' + variable_name

    return variable_name

