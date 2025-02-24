from typing import List



def validate_arrays(a: List[int], b: List[int]) -> bool:

    """Validates if the lengths of two arrays are equal.



    Args:

        a: The first array.

        b: The second array.



    Raises:

        ValueError: If the lengths of the two arrays are not equal.

    """

    if len(a) != len(b):

        raise ValueError(f"Lengths of arrays are not equal: {len(a)} and {len(b)}")

    return True

