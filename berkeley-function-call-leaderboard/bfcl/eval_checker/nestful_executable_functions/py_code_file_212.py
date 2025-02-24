import math



def level(number: float, level: float) -> float:

    """Rounds a floating point number to the nearest multiple of a specified level.



    Args:

        number: The floating point number to be rounded.

        level: The level to which the number should be rounded.



    Raises:

        ValueError: If the level is zero.

    """

    if level == 0:

        raise ValueError("Level cannot be zero")

    levels = number / level

    rounded_levels = math.ceil(levels)

    return rounded_levels * level

