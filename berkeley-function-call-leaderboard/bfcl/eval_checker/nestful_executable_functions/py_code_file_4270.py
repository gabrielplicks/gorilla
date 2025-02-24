def fib(n: int) -> int:

    """Calculates the nth Fibonacci number.



    Args:

        n: The position of the Fibonacci number to calculate.



    Raises:

        TypeError: If the input is not an integer.

    """

    if not isinstance(n, int):

        raise TypeError("Input must be an integer")



    if n == 1 or n == 2:

        return 1

    else:

        return fib(n-1) + fib(n-2)

