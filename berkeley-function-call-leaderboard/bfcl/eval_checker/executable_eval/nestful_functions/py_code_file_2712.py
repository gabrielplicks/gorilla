from typing import List, Tuple



records: List[Tuple[int, float]] = [(10, 2.0), (5, 1.5), (2, 0.5)]



def get_total_wages(records: List[Tuple[int, float]]) -> float:

    """Calculates the total wages for the employees in the records list.



    Args:

        records: A list of tuples, where each tuple contains an employee's ID and their hourly wage.



    Returns:

        The total wages for the employees in the records list.

    """

    total_wages = 0

    for id, wage in records:

        total_wages += id * wage

    return total_wages



assert get_total_wages(records) == 10 * 2.0 + 5 * 1.5 + 2 * 0.5

