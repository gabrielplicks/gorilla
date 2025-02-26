from typing import List



def is_student_name_in_records(student_name: str, records: List[List[str]]) -> bool:

    """Checks if a student's name is in the records.



    Args:

        student_name: The name of the student to search for.

        records: A list of lists containing student names and unique IDs.



    Returns:

        True if the student's name is in the records, False otherwise.

    """

    for sublist in records:

        if student_name in sublist:

            return True

    return False

