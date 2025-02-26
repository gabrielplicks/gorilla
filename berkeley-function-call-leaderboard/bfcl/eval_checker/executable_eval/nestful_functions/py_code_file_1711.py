from typing import List



def extract_heuristic(N: str) -> List[str]:

    """Extracts a heuristic for solving a given problem `N` based on NP-completeness.



    Args:

        N: The problem for which to extract a heuristic.



    Returns:

        A list of heuristics, where each heuristic is an implementation of a heuristic.



    Raises:

        ValueError: If the input `N` is invalid.

    """

    if N == '2-SAT':

        return ['Heuristic1', 'Heuristic2']

    elif N == 'Travelling Salesman Problem':

        return ['Heuristic3', 'Heuristic4']

    elif N == '3-Colorability':

        return ['Heuristic5', 'Heuristic6']

    else:

        raise ValueError('Invalid input!')

