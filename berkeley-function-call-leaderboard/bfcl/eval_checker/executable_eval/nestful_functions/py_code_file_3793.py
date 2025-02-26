from typing import Tuple, List



def reverse_tuple_of_any_length(tuple_input: Tuple) -> List[Tuple]:

    """Returns a list of tuples, where each tuple is the reverse of the input tuple.

    Args:

        tuple_input: The input tuple.

    """

    return [tuple_input[::-1]]

