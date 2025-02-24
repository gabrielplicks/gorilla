from typing import List



def sort_and_add(lst_of_lst: List[List[int]]) -> List[List[int]]:

    """Sorts the elements in each list within a list of lists and returns a new list.



    Args:

        lst_of_lst: A list of lists.



    Returns:

        A new list containing the sorted lists.

    """

    new_list = []

    for lst in lst_of_lst:

        sorted_lst = sorted(lst)

        if sorted_lst:  # Check if the sorted list is not empty

            new_list.append(sorted_lst)

    return new_list

