from typing import Tuple



def calculate_total_apples_oranges(num_apples: int, num_oranges: int, num_days: int) -> Tuple[int, int]:

    """Calculates the total number of apples and oranges eaten in the first `num_days` days.



    Args:

        num_apples: The number of apples eaten per day.

        num_oranges: The number of oranges eaten per day.

        num_days: The number of days.



    Returns:

        A tuple containing the total number of apples and oranges.

    """

    total_apples = 0

    total_oranges = 0



    for day in range(1, num_days + 1):

        total_apples += num_apples

        total_oranges += num_oranges



    return total_apples, total_oranges

