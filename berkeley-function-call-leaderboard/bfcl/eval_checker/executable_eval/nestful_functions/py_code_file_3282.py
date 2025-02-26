from typing import Union



def calculate_score(correct: Union[int, float], incorrect: Union[int, float]) -> int:

    """Calculates the score based on the number of correct and incorrect answers.

    Args:

        correct: The number of correct answers.

        incorrect: The number of incorrect answers.

    """

    if correct >= incorrect:

        return 100

    else:

        return 0

