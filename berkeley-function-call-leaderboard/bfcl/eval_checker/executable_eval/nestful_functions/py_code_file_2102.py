from typing import Optional



def generate_function(num: int) -> Optional[int]:

    """Generates a function that increments a non-negative integer by an offset.



    The offset starts at 1 and increases by 1 for each consecutive call.

    If the input is 0, the function returns None.



    Args:

        num: A non-negative integer.



    Returns:

        The input integer incremented by the offset, or None if the input is 0.

    """

    offset = 0

    if num == 0:

        return None

    else:

        while num >= 0:

            offset += 1

            num += offset

            return num

