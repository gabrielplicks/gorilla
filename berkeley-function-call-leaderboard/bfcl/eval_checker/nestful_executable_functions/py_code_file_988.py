from typing import Dict



def find_student_with_highest_score(student_scores: Dict[str, int]) -> str:

    """Finds the student with the highest score.



    Args:

        student_scores: A dictionary of student names and scores.



    Returns:

        The name of the student with the highest score.

    """

    highest_score = 0

    highest_score_student = None

    for student, score in student_scores.items():

        if score > highest_score:

            highest_score = score

            highest_score_student = student

    return highest_score_student

