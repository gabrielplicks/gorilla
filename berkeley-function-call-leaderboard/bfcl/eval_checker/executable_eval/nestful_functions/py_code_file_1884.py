from typing import List

from collections import Counter



def count_score(scores: List[int], target: int) -> int:

    """Counts the number of scores in a list that are greater than or equal to a target value.



    Args:

        scores: A list of scores.

        target: The target value.



    Returns:

        The number of scores that are greater than or equal to the target value.

    """

    return Counter(score >= target for score in scores).get(True, 0)

