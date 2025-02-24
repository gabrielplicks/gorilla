from typing import Any



def compute_docstring(input_string: str) -> str:

    """Computes the docstring for a function.



    Args:

        input_string: The input string to be formatted as a docstring.



    Returns:

        The computed docstring.

    """

    input_string = input_string.strip()

    docstring = f'''{input_string}'''



    return docstring

