from typing import List, Dict



def filter_employees(employees: List[Dict], gender: str) -> List[Dict]:

    """Filters the employees based on the given rules.



    Args:

        employees: a list of dictionaries representing employees.

        gender: a string representing the desired gender ('male' or 'female').



    Returns:

        a list of dictionaries containing only the filtered employees.

    """

    def predicate(employee: Dict) -> bool:

        return (

            employee['age'] >= 20 and employee['age'] <= 30 and

            employee['gender'] == gender and

            employee['salary'] <= 50000

        )



    filtered_employees = list(filter(predicate, employees))

    return filtered_employees

