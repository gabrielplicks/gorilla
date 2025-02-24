from typing import List, Tuple



def map_scores_to_grades(scores: List[int], conversion_table: Dict[Tuple[int, int], str]) -> List[str]:

    """Maps a list of student scores to a list of letter grades.



    Args:

        scores: A list of student scores.

        conversion_table: A dictionary where the keys are score ranges and the values are letter grades.



    Returns:

        A list of letter grades that correspond to each student's score.

    """

    grades = []

    for score in scores:

        for score_range, grade in conversion_table.items():

            if score_range[0] <= score <= score_range[1]:

                grades.append(grade)

                break

    return grades

