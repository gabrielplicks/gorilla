from typing import List



def indices(bool_list: List[bool]) -> List[int]:

    """Returns the indices of the True elements in the input list.



    Args:

        bool_list: A list of boolean values.



    Returns:

        A list of indices where the corresponding boolean value in the input list is True.

    """

    return [i for i, bool_value in enumerate(bool_list) if bool_value]

