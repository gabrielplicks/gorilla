from typing import List, Tuple



def add_tuples(input: List[Tuple[int, int]]) -> Tuple[int, int]:

    """Adds the elements of each tuple in a list of tuples.



    Args:

        input: A list of tuples.



    Raises:

        ValueError: If the input is not a list of tuples or if the tuples do not have the same length.

    """

    if not isinstance(input, list) or not all(isinstance(t, tuple) for t in input):

        raise ValueError("Input must be a list of tuples")

    tuple_length = len(input[0])

    if not all(len(t) == tuple_length for t in input):

        raise ValueError("All tuples must have the same length")

    return tuple(sum(t[i] for t in input) for i in range(tuple_length))

