from collections import Counter

from typing import List



def get_prior(labels: List[str], target: str) -> float:

    """Calculates the prior probability of the target class using the class labels' frequencies.



    Args:

        labels: A list of class labels.

        target: The target class.



    Returns:

        The prior probability of the target class as a decimal value between 0 to 1.

    """

    frequencies = Counter(labels)

    total = len(labels)

    return frequencies[target] / total

