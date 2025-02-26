from typing import Tuple



def calculate_fnr(tn: int, fn: int, p: int) -> float:

    """Calculates the False Negative Rate (FNR) given the number of true negatives (TN), false negatives (FN), and the number of total positives (P).



    Args:

        tn: The number of true negatives.

        fn: The number of false negatives.

        p: The number of total positives.



    Returns:

        The False Negative Rate (FNR).

    """

    fnr = fn / p

    return fnr

