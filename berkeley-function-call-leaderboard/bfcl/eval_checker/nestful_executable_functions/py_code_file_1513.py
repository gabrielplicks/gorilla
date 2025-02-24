from typing import Union



def generate_fstring(is_special: bool) -> str:

    """Returns a string with a f-string containing a value that is either 100 or 500, depending on whether the `is_special` parameter is `True` or `False`.



    Args:

        is_special: A boolean value that determines the value of the f-string.



    Returns:

        A string with a f-string containing a value that is either 100 or 500, depending on whether the `is_special` parameter is `True` or `False`.

    """

    if is_special:

        value = 500

    else:

        value = 100



    return f'{value:.2f}'

