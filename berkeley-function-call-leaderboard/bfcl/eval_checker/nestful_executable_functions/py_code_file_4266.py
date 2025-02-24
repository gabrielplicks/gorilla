from typing import List, Tuple, Union



def find_index_and_value(lst: List[int], value: int) -> Union[Tuple[int, int], None]:

    """Finds the first occurrence of a target value in a list of integers and returns a tuple containing both the index and the value.



    Args:

        lst: A list of integers.

        value: The target value to find in the list.



    Returns:

        A tuple containing the index and value of the first occurrence of the target value in the list, or `None` if the value is not found.

    """

    for i, item in enumerate(lst):

        if item == value:

            return i, item

    return None

