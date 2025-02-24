from typing import Union



def concat_ignore_trailing_zeros(a: Union[int, str], b: Union[int, str]) -> str:

    """Concatenates two inputs, `a` and `b`, into a single string, ignoring any trailing zeros in the second input, `b`.



    Args:

        a: The first input.

        b: The second input.



    Returns:

        The concatenated string.

    """

    a_str = str(a)

    b_str = str(b).strip('0')

    return a_str + b_str

