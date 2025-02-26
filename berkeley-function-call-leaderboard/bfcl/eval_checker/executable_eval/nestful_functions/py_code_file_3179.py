from typing import List, Tuple, Union



def get_scores(grades: List[Tuple[int, str, float]], student_id: int) -> Union[List[float], None]:

    """

    Returns a list of scores associated with the given student id. If the student is not present

    in the grades list, returns None. The grades list is a list of tuples in the format

    (student_id, subject, score), where student_id is an integer, subject is a string, and

    score is a float.



    Args:

        grades: A list of tuples representing grades.

        student_id: An integer representing a student's ID.

    """

    filtered_grades = [grade for grade in grades if grade[0] == student_id]



    if not filtered_grades:

        return None



    scores = [grade[2] for grade in filtered_grades]

    return scores

