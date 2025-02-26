def clip(x: float, min_val: float, max_val: float) -> float:

    """Clips a number `x` to be at least `min_val` and at most `max_val`.



    Args:

        x: The number to clip.

        min_val: The minimum value that `x` can take.

        max_val: The maximum value that `x` can take.

    """

    return min(max(x, min_val), max_val)

