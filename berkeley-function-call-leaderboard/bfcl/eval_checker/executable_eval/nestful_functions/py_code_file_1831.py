from typing import Dict, Tuple, List



def compute_tuple_frequency(tuple_list: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:

    """Computes the frequency of each tuple in the list.



    Args:

        tuple_list: A list of tuples.



    Returns:

        A dictionary where the key is the tuple and the value is the frequency of that tuple.

        If a tuple does not exist in the list, its frequency is 0.

    """

    tuple_frequency: Dict[Tuple[int, int], int] = {}

    for t in tuple_list:

        if t in tuple_frequency:

            tuple_frequency[t] += 1

        else:

            tuple_frequency[t] = 1

    return tuple_frequency

