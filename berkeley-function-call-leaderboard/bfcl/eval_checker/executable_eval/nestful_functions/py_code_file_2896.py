from typing import Union



def percentage_grade(score: Union[int, float]) -> str:

    """Calculates the percentage grade based on the given score.



    Args:

        score: The score to calculate the percentage grade for.



    Returns:

        A string representing the percentage grade.

    """

    if not isinstance(score, (int, float)) or score < 0 or score > 100:

        return 'Invalid input'

    elif score >= 90:

        return 'A'

    elif score >= 80:

        return 'B'

    elif score >= 70:

        return 'C'

    elif score >= 60:

        return 'D'

    else:

        return 'F'

