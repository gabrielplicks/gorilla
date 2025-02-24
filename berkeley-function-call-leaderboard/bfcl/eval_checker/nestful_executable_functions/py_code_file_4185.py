from typing import List, Tuple



def get_max_by_value(tuple_list: List[Tuple[int, int]]) -> Tuple[int, int]:

    """Returns the tuple with the highest second value from the list of tuples.

    If there are multiple tuples with the same highest second value,

    returns the one with the lowest first value.



    Args:

        tuple_list: A list of tuples.

    """

    sorted_tuples = sorted(tuple_list, key=lambda x: x[1], reverse=True)

    lowest_first_value_tuple = None

    for tup in sorted_tuples:

        if not lowest_first_value_tuple or tup[0] < lowest_first_value_tuple[0]:

            lowest_first_value_tuple = tup

            break



    return lowest_first_value_tuple

