from typing import List, Dict



def find_students(student_records: List[Dict], grade: int) -> List[Dict]:

    """Finds all student records in a list that have a specific grade value.



    Args:

        student_records: A list of student records. Each student record is a dictionary.

        grade: The grade value to search for.



    Returns:

        A list of all student records that have the specified grade value.

    """

    matched_students = []



    for student in student_records:

        for key, value in student.items():

            if key == 'grade' and value == grade:

                matched_students.append(student)

                break



    return matched_students

