from typing import List, Tuple



def conditional_probability(occurrences: List[Tuple[bool, bool]]) -> float:

    """Calculates the conditional probability of A given B.



    Args:

        occurrences: A list of tuples, where each tuple represents the occurrence of A and B events.



    Returns:

        The conditional probability of A given B as a float.

    """

    a_count = sum(a for a, b in occurrences)

    b_count = sum(b for a, b in occurrences)

    joint_probability = sum(a & b for a, b in occurrences)

    conditional_probability = joint_probability / b_count



    return conditional_probability

