from typing import Union



def get_thing_string(num: Union[int, float]) -> str:

    """Returns a string of the form "N [thing]" based on the input number.

    Args:

        num: The input number.

    """

    if num == 1:

        return f"{num} thing"

    elif num > 1:

        return f"{num} things"

    else:

        return "0 things"

