from typing import Dict, Union



def assign_exam_score(result: Union[int, float]) -> int:

    """Assigns a score to an exam based on a student's exam result.

    Args:

        result: The result of the exam, where 0 indicates a failed exam and 1 indicates a passed exam.

    Returns:

        The score of the exam, which is an integer between 0 and 100.

    """

    score_mapping: Dict[Union[int, float], int] = {0: 0, 1: 100}

    if result not in score_mapping:

        return 0  # Return 0 for any result that is not present in the mapping

    return score_mapping[result]

