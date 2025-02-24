from typing import Literal



def check_whether_the_student_has_the_permission_to_skip_the_class(student_name: str, student_age: int) -> bool:

    """Checks whether the student has the permission to skip the class based on the given conditions.

    Args:

        student_name: The name of the student.

        student_age: The age of the student.

    """

    if student_name.lower().startswith("a") or student_name.lower().startswith("e") or student_name.lower().startswith("i") or student_name.lower().startswith("o") or student_name.lower().startswith("u"):

        if 6 <= student_age <= 12 or 18 <= student_age <= 21:

            return True

    else:

        if 13 <= student_age <= 17:

            return True

    return False

