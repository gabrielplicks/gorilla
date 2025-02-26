from typing import Dict, List



def dict_sum_matrix(d: Dict[str, List[float]]) -> List[List[float]]:

    """Creates a matrix of size (10, 10) where the element at index [i, j] is the sum of d['x'][i] and d['y'][j].



    Args:

        d: A dictionary with keys 'x' and 'y' that map to lists of length 10.



    Returns:

        A matrix of size (10, 10) containing the sum of d['x'][i] and d['y'][j].

    """

    matrix = [[d['x'][i] + d['y'][j] for j in range(10)] for i in range(10)]

    return matrix

