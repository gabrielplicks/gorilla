def custom_round(x: float, factor: float) -> float:

    """Rounds a given numeric value to the closest multiple of a specified factor.



    Args:

        x: The numeric value to round.

        factor: The factor to round to.



    Returns:

        The rounded value.



    Raises:

        ValueError: If the factor is not a positive number.

    """

    if factor <= 0:

        raise ValueError("The factor must be a positive number.")

    return round(x / factor) * factor

