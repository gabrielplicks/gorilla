from typing import List



def select_first_and_last(rectangular_data: List[List[float]]) -> List[List[float]]:

    """Selects the first and last elements of each sub-list in a list of lists.



    Args:

        rectangular_data: A list of lists representing rectangular data.



    Returns:

        A list of lists containing just the first and last elements of each sub-list.

    """

    new_data = []

    for sublist in rectangular_data:

        first = sublist[0]

        last = sublist[-1]

        new_data.append([first, last])

    return new_data

