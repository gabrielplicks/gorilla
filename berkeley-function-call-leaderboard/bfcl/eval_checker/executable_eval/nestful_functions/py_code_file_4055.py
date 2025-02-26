from typing import List



def remove_duplicate_pairs(pairs: List[List[int]]) -> List[List[int]]:

    """

    Removes duplicate pairs from a list of pairs.

    A pair is considered a duplicate if it contains the same elements as another pair,

    but in a different order.



    Args:

        pairs: A list of pairs. Each pair is a 2-element list of integers.



    Returns:

        A new list of unique pairs.

    """

    unique_pairs = []

    for pair in pairs:

        if pair[::-1] in unique_pairs:

            continue

        unique_pairs.append(pair)

    return unique_pairs

