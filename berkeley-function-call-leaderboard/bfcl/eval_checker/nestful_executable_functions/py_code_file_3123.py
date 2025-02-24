from typing import List, Tuple



def sparse_array_to_dict(sparse_array: List[Tuple[int, int]]) -> Dict[int, int]:

    """Converts a sparse array of tuples into a dictionary.



    Args:

        sparse_array: A sparse array of tuples. Each tuple contains two elements, the first being the index and the second being the element value.



    Returns:

        A dictionary where the keys are the indices and the values are the corresponding elements.

    """

    dict = {}

    for index, value in sparse_array:

        dict[index] = value

    return dict

