from typing import List



def select_first_element_of_each_list(lst: List[List[int]]) -> List[int]:

    """Selects the first element of each list in a list of lists.



    Args:

        lst: A list of lists.



    Raises:

        TypeError: If the input is not a list.

        ValueError: If the input is not a list of lists.

    """

    if not isinstance(lst, list):

        raise TypeError('Input must be a list')

    if not all(isinstance(l, list) for l in lst):

        raise ValueError('Input must be a list of lists')

    return [l[0] for l in lst if l]

