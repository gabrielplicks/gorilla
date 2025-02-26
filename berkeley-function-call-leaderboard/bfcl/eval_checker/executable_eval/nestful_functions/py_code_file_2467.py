from typing import Dict



def normalize_counts(counts: Dict[str, float]) -> Dict[str, float]:

    """Normalizes a dictionary of counts to sum up to 1.



    Args:

        counts: A dictionary with strings as keys and counts as values.



    Returns:

        A dictionary with the same keys as the input dictionary and normalized

        counts as values.

    """

    tree = {}

    total = 0

    for key, value in counts.items():

        tree[key] = value

        total += value

    for key in tree:

        tree[key] = tree[key] / total



    return tree

