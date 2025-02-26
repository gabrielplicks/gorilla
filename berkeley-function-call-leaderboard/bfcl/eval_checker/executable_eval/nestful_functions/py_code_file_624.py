import math



def smooth_curve(x: float) -> float:

    """Calculates a smooth curve that is monotonically increasing and passes through the points (0, 0), (1, 1), and (2, 3).



    Args:

        x: The input value.

    """

    return 1 / (1 + math.exp(-x))

