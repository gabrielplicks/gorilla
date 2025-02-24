from typing import List, Optional



def find_lowest_temperature(T: List[float]) -> Optional[float]:

    """Finds the lowest temperature from a list of temperatures.



    Args:

        T: A list of temperatures.



    Returns:

        The lowest temperature, or None if the list is empty.

    """

    if len(T) == 0:

        return None

    lowest_temperature = T[0]

    for temperature in T[1:]:

        if temperature < lowest_temperature:

            lowest_temperature = temperature

    return lowest_temperature

