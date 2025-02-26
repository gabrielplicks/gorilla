import json

from typing import List



class Employee:

    def __init__(self, name: str, salary: float):

        self.name = name

        self.salary = salary



class EmployeeJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, Employee):

            return {"name": obj.name, "salary": obj.salary}

        return super().default(obj)



def serialize_employees_to_json(employees: List[Employee]) -> str:

    """Serializes a list of Employee objects into a JSON-formatted string.



    Args:

        employees: A list of Employee objects.



    Returns:

        A JSON-formatted string representing the list of Employee objects.

    """

    json_string = json.dumps(employees, cls=EmployeeJSONEncoder)

    return json_string

