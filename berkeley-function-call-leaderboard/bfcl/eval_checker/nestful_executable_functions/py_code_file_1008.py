from typing import Union



def velocity(time: Union[int, float], acceleration: Union[int, float] = 9.81) -> float:

    """Calculates the velocity of a falling object.

    Args:

        time: The time elapsed.

        acceleration: The acceleration of the object. Defaults to 9.81.

    """

    return acceleration * time

