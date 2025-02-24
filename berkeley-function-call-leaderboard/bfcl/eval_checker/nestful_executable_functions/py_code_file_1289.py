from typing import List



def process_vector(v: List[float], s: float, do_normalize: bool) -> List[float]:

    """Processes a vector based on the given scalar and boolean.



    If `do_normalize` is True, the function returns a normalized vector.

    Otherwise, the function returns a vector whose elements are the square of the difference of `v` and `s`.



    Args:

        v: The input vector.

        s: The scalar.

        do_normalize: A boolean indicating whether to normalize the vector or not.

    """

    if do_normalize:

        return [element / len(v) for element in v]

    else:

        return [pow(element - s, 2) for element in v]

