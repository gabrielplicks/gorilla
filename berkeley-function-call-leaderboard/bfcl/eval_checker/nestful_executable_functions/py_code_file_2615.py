from typing import Tuple



def startup_fuel(capacity: int, time: int) -> float:

    """Calculates startup fuel based on time-dependent startup capacity.



    Args:

        capacity: The startup capacity.

        time: The time of day.



    Returns:

        The startup fuel.

    """

    m = 10

    b = 0

    return m * capacity + b * time

