from typing import List



def remove_integer(input_list: List[int], parameter: int) -> List[int]:

    """Removes all instances of a specified parameter from a list of integers.



    Args:

        input_list: The list of integers.

        parameter: The integer to remove from the list.



    Returns:

        The updated list with the specified parameter removed.

    """

    assert all(isinstance(item, int) for item in input_list), "Input must be a list of integers"

    return [i for i in input_list if i != parameter]

