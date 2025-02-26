from typing import List



def my_concat(list1: List[int], list2: List[int]) -> List[int]:

    """Concatenates two lists together.



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        A new list that contains the elements of the two input lists concatenated together.

    """

    new_list = []

    for element in list1:

        new_list.append(element)

    for element in list2:

        new_list.append(element)

    return new_list

