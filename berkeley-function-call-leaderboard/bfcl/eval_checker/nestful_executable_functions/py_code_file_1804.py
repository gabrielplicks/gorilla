from typing import List



def normalize_list(integers: List[int]) -> List[float]:

    """Normalizes a list of integers to a list of floats between 0 and 1.



    Args:

        integers: The list of integers to normalize.



    Returns:

        A list of floats between 0 and 1.

    """

    max_value = max(integers)

    return [num / max_value for num in integers]

