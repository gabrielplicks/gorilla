from typing import List, Dict



def get_employees_by_department(my_list: List[Dict], departments: List[str]) -> List[str]:

    """

    Returns a list of employees' names from `my_list` whose departments match the ones in `departments`.



    Args:

        my_list: A list of dictionaries, where each dictionary represents an employee and has 'id', 'name', and 'department' keys.

        departments: A list of departments.

    """

    employees = [my_dict['name'] for my_dict in my_list if my_dict['department'] in departments]

    return employees

