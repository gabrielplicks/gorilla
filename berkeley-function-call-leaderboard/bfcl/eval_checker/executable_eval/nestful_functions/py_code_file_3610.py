from typing import Union



def xor_or(a: Union[bool, int], b: Union[bool, int], c: Union[bool, int] = False) -> bool:

    """Calculates the result of (a xor b) or c, where xor means exclusive OR.

    Args:

        a: The first boolean argument.

        b: The second boolean argument.

        c: The third boolean argument. Defaults to False.

    """

    if not isinstance(a, bool) or not isinstance(b, bool) or not isinstance(c, bool):

        raise ValueError("All arguments must be boolean values.")

    xor = (a or b) and not (a and b)

    return xor or c

