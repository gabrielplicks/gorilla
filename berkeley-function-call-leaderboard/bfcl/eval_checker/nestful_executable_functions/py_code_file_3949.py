from typing import List, Tuple



def get_top_two_scores_and_names(scores_and_names: List[Tuple[str, float]]) -> List[Tuple[str, float]]:

    """

    Returns a list of tuples containing the first two highest scores along with their corresponding names.

    Args:

        scores_and_names: A list of tuples containing names and scores.

    """

    sorted_scores_and_names = sorted(scores_and_names, key=lambda x: x[1], reverse=True)

    top_two_scores_and_names = sorted_scores_and_names[:2]

    return top_two_scores_and_names

