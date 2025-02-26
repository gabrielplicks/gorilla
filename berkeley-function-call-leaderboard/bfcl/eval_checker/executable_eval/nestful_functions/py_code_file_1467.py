import itertools

from typing import List



def split_on_special(data: List[int], x: int) -> List[List[int]]:

    """Splits a list of integers into sequences based on a special element.



    Args:

        data: The list of integers.

        x: The special element that signals the start of a new sequence.



    Returns:

        A list of sequences, where each sequence corresponds to all the elements

        up to and including the next special element.

    """

    groups = itertools.groupby(data, lambda y: y == x)

    sequences = []

    for key, group in groups:

        if key:

            continue

        else:

            sequences.append(list(group))



    return sequences

