from typing import List



def find_all_true_indices(input_list: List[bool]) -> List[int]:

    """Finds all indices in the input list where the value is True.



    Args:

        input_list: A list of boolean values.



    Returns:

        A list of indices where the value is True.

    """

    return [i for i, value in enumerate(input_list) if value]

