from typing import Dict



def compute_rank(num: int) -> int:

    """

    Computes the rank of an integer number.

    The rank is the number of times a digit occurs in a number.

    For example, the rank of 12345 is 5, since each digit occurs once.

    Args:

        num: The integer number to compute the rank of.

    """

    digit_counts: Dict[str, int] = {}

    for digit in str(num):

        digit_counts[digit] = digit_counts.get(digit, 0) + 1

    rank = sum(digit_counts.values())

    return rank

