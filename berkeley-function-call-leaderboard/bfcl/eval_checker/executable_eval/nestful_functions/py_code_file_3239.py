from typing import List, Dict



def generate_messages(classrooms: List[Dict[str, str]]) -> List[str]:

    """Generates messages for each classroom in the list.



    Args:

        classrooms: A list of dictionaries containing classroom information.



    Returns:

        A list of messages.

    """

    messages = []

    for classroom in classrooms:

        name = classroom['name']

        students = classroom['students']

        num_students = len(students)

        message = f"There are {num_students} students in the classroom named {name}."

        messages.append(message)

    return messages

