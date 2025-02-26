import json

from typing import List, Dict



def read_employee_records(path: str) -> List[Dict[str, str]]:

    """Reads information from a JSON file and returns a list of dictionaries.

    Each dictionary contains the `id` and `department` fields of an employee record.

    The function only includes records where the `department` field's value is not `None`.



    Args:

        path: The path to the JSON file containing employee records.



    Returns:

        A list of dictionaries containing the `id` and `department` fields of employee records.

    """

    with open(path, 'r') as f:

        data = json.load(f)



    employee_records = []

    for record in data:

        if record['department'] is not None:

            employee_records.append({'id': record['id'], 'department': record['department']})



    return employee_records

