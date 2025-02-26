from collections import Counter

from typing import Dict, List



def compute_probabilities(numbers: List[int]) -> Dict[int, float]:

    """Computes the probability of each number occurring in a list.



    The probability is calculated by dividing the occurrence of each number by the length of the list.

    The function handles the case when any number occurs more than once, and handles the case of a zero length list by returning an empty dictionary.

    The function never raises an exception.



    Args:

        numbers: The list of numbers.



    Returns:

        A dictionary where the keys are the numbers and the values are their probabilities.

    """

    try:

        count = Counter(numbers)

        return {k: v / len(numbers) for k, v in count.items()}

    except ZeroDivisionError:

        return {}

