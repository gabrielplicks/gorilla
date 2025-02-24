from typing import List



def constant_feature_matching_score(x: List[str], y: List[str]) -> int:

    """Computes the constant feature matching score for two sets of features x and y.

    The constant feature matching score is defined as the number of features present in both x and y.

    Args:

        x: A list of features.

        y: A list of features.

    """

    score = 0

    for feature in x:

        if feature in y:

            score += 1

    return score

