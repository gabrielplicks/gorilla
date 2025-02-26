from typing import Union



def convert_percentage_to_score(percentage: float) -> int:

    """Converts a percentage value between 0 and 1 to a score between 0 and 5.



    Args:

        percentage: The percentage value to convert.



    Returns:

        The corresponding score.

    """

    return int(round(percentage * 5))



def convert_score_to_percentage(score: Union[int, float]) -> float:

    """Converts a score between 0 and 5 to a percentage value between 0 and 1.



    Args:

        score: The score to convert.



    Returns:

        The corresponding percentage value.

    """

    return score / 5

