from typing import List, Dict



def build_average_grades_dict(student_list: List[Dict[str, any]]) -> Dict[str, float]:

    """Builds a dictionary containing the name of a student as the key and their average grade as the value.



    Args:

        student_list: A list of dictionaries, where each dictionary represents a student, and the key-value pairs represent the student's name and the grades they received for each course.



    Returns:

        A dictionary containing the average grades for each student.

    """

    average_grades = {}

    for student in student_list:

        name = student['name']

        grades = sum(student['grades']) / len(student['grades'])

        average_grades[name] = grades



    return average_grades

