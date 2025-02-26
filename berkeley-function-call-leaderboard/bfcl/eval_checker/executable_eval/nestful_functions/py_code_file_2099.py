from typing import Optional



class Employee:

    def __init__(self, name: str):

        self.name = name



    def get_name(self) -> str:

        return self.name



    def set_name(self, name: str):

        self.name = name



def change_employee_name(employee: Employee) -> None:

    """Changes the employee's name to "" if it isn't already and displays the new name.



    Args:

        employee: The Employee object.

    """

    if employee.get_name() != "":

        employee.set_name("")

    print(employee.get_name())

