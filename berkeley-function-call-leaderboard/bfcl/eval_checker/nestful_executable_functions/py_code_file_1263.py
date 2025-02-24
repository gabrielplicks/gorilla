def multiply_and_check(a: int, b: int) -> bool:

    """

    Multiply two integers and return True if the result is even.



    Args:

        a: The first integer.

        b: The second integer.



    Returns:

        True if the result is even, False otherwise.

    """

    result = a * b

    return (result & 1) == 0

