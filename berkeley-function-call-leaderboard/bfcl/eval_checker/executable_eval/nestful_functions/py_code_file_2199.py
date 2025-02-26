from typing import Tuple



def validate_velocities(v1: float, v2: float, v3: float, v4: float) -> bool:

    """Validates whether the four velocities are in the correct order.



    Args:

        v1: The first velocity.

        v2: The second velocity.

        v3: The third velocity.

        v4: The fourth velocity.



    Returns:

        True if the velocities are in the correct order, False otherwise.

    """

    if v1 <= 0 or v2 <= 0 or v3 <= 0 or v4 <= 0:

        return False

    if v1 < v2 < v3 < v4:

        return True

    else:

        return False

