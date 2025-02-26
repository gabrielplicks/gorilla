from typing import List



def format_pairs(pairs: List[List[int]]) -> List[str]:

    """Formats a list of pairs of integers into a list of strings.



    Args:

        pairs: A list of pairs of integers.



    Returns:

        A list of strings formatted as '<', '>', or '=' based on the comparison of the integers in each pair.

    """

    return [f'{a} {"<" if a < b else ">" if a > b else "="} {b}' for a, b in pairs]

