from typing import List, Dict



def highest_salary_employee(employees: List[Dict[str, str]]) -> str:

    """Returns the name of the employee with the highest salary whose title is not "CEO".

    If there is a tie for highest salary, returns the first employee with that salary.

    If there are no employees with the desired title, returns `None`.

    Args:

        employees: A list of dictionaries containing employee information.

    """

    highest_salary = -1

    highest_salary_employee = None

    for employee in employees:

        if employee['title'] == 'CEO':

            continue

        if employee['salary'] > highest_salary:

            highest_salary = employee['salary']

            highest_salary_employee = employee['name']

    return highest_salary_employee

