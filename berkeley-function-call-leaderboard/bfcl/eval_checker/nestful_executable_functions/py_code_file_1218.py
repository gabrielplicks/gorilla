from typing import List, Tuple



def reformat_student_records(records: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:

    """Reformats a list of student records by swapping the first and last names.



    Args:

        records: A list of tuples representing student records. Each tuple contains the student's first name, last name, and ID number.



    Returns:

        A new list of tuples with the first and last names swapped.

    """

    return [(student[1], student[0], student[2]) for student in records]

