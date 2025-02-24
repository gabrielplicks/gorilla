from typing import Union



def handle_none(value: Union[str, None]) -> str:

    """Returns a new string that is the input argument's value with a default value of 'No value' if input is None.



    Args:

        value: The input value.

    """

    return value or 'No value'

