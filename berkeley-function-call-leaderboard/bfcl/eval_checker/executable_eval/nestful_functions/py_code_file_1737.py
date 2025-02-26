from typing import List, Union



def clip_list(l: List[Union[int, float]], min_val: float, max_val: float) -> List[float]:

    """Clips the values in a list to a specified range.



    Args:

        l: The list of values to clip.

        min_val: The minimum value of the range.

        max_val: The maximum value of the range.

    """

    return [clip(value, min_val, max_val) for value in l]

