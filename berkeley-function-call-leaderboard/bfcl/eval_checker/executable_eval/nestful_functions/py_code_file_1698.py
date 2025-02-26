from typing import Union



def concatenate_with_space(x: str, y: str) -> Union[str, None]:

    """Concatenates two strings `x` and `y` with a space in between.

    If one of the strings is empty, returns the other string instead.

    Uses string interpolation to format the return value.



    Args:

        x: The first string.

        y: The second string.

    """

    if len(x) == 0:

        return y

    if len(y) == 0:

        return x

    return f"{x} {y}"

