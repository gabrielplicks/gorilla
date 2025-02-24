def fibonacci_recursive(n: int) -> int:

    """Calculates the nth number in the Fibonacci sequence using recursion.



    Args:

        n: The position of the number in the sequence.



    Returns:

        The nth number in the Fibonacci sequence.

    """

    if n <= 1:

        return n

    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

