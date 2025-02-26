from typing import List



def get_middle_elements(lst: List[int]) -> List[int]:

    """Returns a new list containing the middle two elements of the input list.



    If there are an odd number of elements, the middle element should be included in the returned list.

    If there are two middle elements, the function should return a list containing both elements.

    If the input list is empty, the function should return an empty list.



    Args:

        lst: A list of integers.



    Returns:

        A new list containing the middle elements of the input list.

    """

    if not lst:

        return []

    if len(lst) == 1:

        return lst

    middle_index = (len(lst) - 1) // 2



    if len(lst) % 2 == 1:

        return lst[middle_index:middle_index + 1]

    else:

        return lst[middle_index:middle_index + 2]

