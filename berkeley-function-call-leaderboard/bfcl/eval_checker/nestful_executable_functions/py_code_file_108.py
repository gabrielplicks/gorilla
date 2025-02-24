from typing import List, Dict



def get_average_GPA(students: List[Dict[str, float]]) -> List[float]:

    """Returns a list of the average GPA of all students who have a `GPA` attribute.



    Args:

        students: A list of dictionaries, where each dictionary represents a student and has a `GPA` attribute.



    Returns:

        A new list containing the `GPA` of every student in the input list.

    """

    return [student['GPA'] for student in students if 'GPA' in student]

