from typing import Optional



def calculate_vertical_velocity(v_0: Optional[float], a: Optional[float], t: Optional[float]) -> Optional[float]:

    """Calculates the vertical velocity of an object after a given time.



    Args:

        v_0: The initial velocity of the object.

        a: The acceleration of the object.

        t: The time elapsed.



    Returns:

        The vertical velocity of the object, or None if any of the parameters is None.

    """

    if v_0 is None or a is None or t is None:

        return None

    return v_0 + a * t

