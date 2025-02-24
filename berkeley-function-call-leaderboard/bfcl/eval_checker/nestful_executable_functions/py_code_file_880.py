from typing import List, Dict



def process_students(records: List[Dict[str, object]]) -> List[Dict[str, object]]:

    """Processes a list of student records and returns a new list of dictionaries containing the name and the number of assignments completed.



    Args:

        records: A list of student records. Each record is a dictionary containing the student's name and the number of assignments completed.



    Returns:

        A new list of dictionaries containing the name and the number of assignments completed.

    """

    result = []

    for record in records:

        if record.get('assignments') is not None:

            if isinstance(record['assignments'], int):

                result.append({

                    'name': record['name'],

                    'assignments': record['assignments']

                })

    return result

