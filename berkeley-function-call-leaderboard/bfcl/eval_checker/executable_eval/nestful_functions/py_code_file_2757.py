from typing import List



def match_pairs(a: List[List[str]], b: List[List[str]]) -> List[List[str]]:

    """Creates a new list that includes only the key-value pairs that match exactly in both `a` and `b`.



    Args:

        a: A list of key-value pairs.

        b: A list of key-value pairs.



    Returns:

        A list of key-value pairs that match exactly in both `a` and `b`.

    """

    result = []



    for pair_a in a:

        for pair_b in b:

            if pair_a[0] == pair_b[0] and pair_a[1] == pair_b[1]:

                result.append(pair_a)



    return result

