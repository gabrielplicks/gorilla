from typing import List



def scale_linearly(lst: List[float], scale: float) -> List[float]:

    """Scales a list of numbers linearly by a given scale value.

    Args:

        lst: The list of numbers to be scaled.

        scale: The scale value by which each number in the list is multiplied.

    """

    if scale == 0:

        return []

    if len(lst) == 1:

        return [lst[0] * scale]

    scaled_lst = []

    for val in lst:

        scaled_lst.append(val * scale)

    if scale & (scale - 1) == 0:  # If scale is a power of two

        return [int(val) for val in scaled_lst]  # Return a list of integers

    else:

        return scaled_lst  # Return a list of floats

