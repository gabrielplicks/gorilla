from typing import List, Union



def count_elements_in_nested_list(nested_list: List[Union[int, List[int]]], x: int) -> int:

    """

    Counts the number of elements in a nested list that are equal to a given value x.



    Args:

        nested_list: A list of lists containing integers.

        x: The value to count in the nested list.

    """

    count = 0



    for item in nested_list:

        if isinstance(item, list):

            count += count_elements_in_nested_list(item, x)

        elif item == x:

            count += 1



    return count

