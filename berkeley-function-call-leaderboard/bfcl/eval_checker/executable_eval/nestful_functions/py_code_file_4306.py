from typing import Dict, List



def get_element_counts(lst: List[int]) -> Dict[int, int]:

    """Generates a dictionary with the keys as the elements of the list and the values as the number of occurrences of the elements in the list.



    Args:

        lst: The list of elements.



    Returns:

        A dictionary with the keys as the elements of the list and the values as the number of occurrences of the elements in the list.

    """

    elements_count = {}

    for element in lst:

        if element in elements_count:

            elements_count[element] += 1

        else:

            elements_count[element] = 1

    return elements_count

