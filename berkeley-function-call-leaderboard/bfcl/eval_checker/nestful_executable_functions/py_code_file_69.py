from typing import Dict



def dict_to_probabilities(counts: Dict[str, int]) -> Dict[str, float]:

    """Calculates the probabilities of each key in a dictionary of counts.



    Args:

        counts: A dictionary of counts.



    Returns:

        A dictionary of probabilities.

    """

    sum_counts = sum(counts.values())



    if sum_counts == 0:

        n_keys = len(counts)

        probabilities = {key: 1 / n_keys for key in counts.keys()}

    else:

        probabilities = {key: value / sum_counts for key, value in counts.items()}



    return probabilities

