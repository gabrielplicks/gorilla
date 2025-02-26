from typing import Tuple



def invert_padding(padding: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:

    """Inverts the padding values in a tuple.



    Args:

        padding: A tuple of padding values in the order (top, right, bottom, left).



    Returns:

        A tuple of inverted padding values in the same order as the input.

    """

    assert len(padding) == 4, "Padding must be a tuple of length 4"

    return tuple(-x for x in padding)

