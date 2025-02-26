def multiply_by_two_or_three(input_value: int, times_two: bool) -> int:

    """

    Multiplies the input value by either two or three, depending on the value of `times_two`.

    If `times_two` is `True`, the value is multiplied by two. If it is `False`, the value is

    multiplied by three.



    Args:

        input_value: The input value to be multiplied.

        times_two: A boolean value indicating whether to multiply by two or three.



    Raises:

        ValueError: If the input value is less than zero.

    """

    if input_value < 0:

        raise ValueError("Expected non-negative input")



    if times_two:

        return input_value * 2

    else:

        return input_value * 3

