from typing import List, Tuple



def highest_paid_employee(employees: List[Tuple[str, float]]) -> str:

    """Returns the name of the highest paid employee from a list of employees.



    Args:

        employees: A list of employees, where each employee is represented as a tuple containing their name and salary.



    """

    highest_salary = 0

    highest_paid_employee = ""



    for employee in employees:

        name, salary = employee



        if salary > highest_salary:

            highest_salary = salary

            highest_paid_employee = name



    return highest_paid_employee

