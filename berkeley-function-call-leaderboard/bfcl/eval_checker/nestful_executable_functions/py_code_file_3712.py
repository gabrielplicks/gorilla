def fibonacci_using_while_loop(n: int) -> int:

    """Computes the nth number in the Fibonacci sequence using a while loop and variable assignment.



    Args:

        n: A positive integer representing the position of the desired Fibonacci number.



    Returns:

        The nth number in the Fibonacci sequence.

    """

    if n <= 0:

        return 0

    if n == 1:

        return 1



    a = 0

    b = 1

    while n > 0:

        a += b

        b = a - b

        n -= 1



    return a

