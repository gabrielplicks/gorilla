from typing import List, Tuple



def capitalize_first_names(employee_info: List[Tuple[str, str]]) -> List[str]:

    """

    Returns a list of all the employees' first names with the first letter capitalized.



    Args:

        employee_info: A list of tuples representing employee information. Each tuple contains the first name and last name of an employee.



    Returns:

        A list of capitalized first names.

    """

    return [name.capitalize() for name, _ in employee_info]

