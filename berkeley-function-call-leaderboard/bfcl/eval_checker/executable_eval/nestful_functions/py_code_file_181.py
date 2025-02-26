from typing import List



def bool_array_to_int_array(bool_array: List[bool], value_true: int = 1, value_false: int = 0) -> List[int]:

    """Converts an array of booleans to an array of integers.



    Args:

        bool_array: An array of booleans.

        value_true: The integer value to be used for `True`.

        value_false: The integer value to be used for `False`.



    Returns:

        An array of integers, where each boolean value in `bool_array` is converted to its respective integer value.

    """

    return [value_true if b else value_false for b in bool_array]

