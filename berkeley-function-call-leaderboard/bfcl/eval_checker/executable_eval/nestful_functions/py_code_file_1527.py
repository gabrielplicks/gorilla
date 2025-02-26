from typing import List



def get_longest_unique_sublist(list_of_elements: List[int]) -> List[int]:

    """Returns the longest possible sublist of unique elements from the input list.



    Args:

        list_of_elements: A list of elements.



    Returns:

        The longest possible sublist of unique elements.

    """

    unique_sublist = []



    for element in list_of_elements:

        if element not in unique_sublist:

            unique_sublist.append(element)

        else:

            break  # Stop if a duplicate element is found



    return unique_sublist

