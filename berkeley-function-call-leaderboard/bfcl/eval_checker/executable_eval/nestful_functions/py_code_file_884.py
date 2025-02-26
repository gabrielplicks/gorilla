from typing import List



def convert_to_set_then_back_to_list(numbers: List[int]) -> List[int]:

    """Converts a list of numbers to a set, and then back to a list again.



    Args:

        numbers: A list of numbers.

    """

    return list(set(numbers))

