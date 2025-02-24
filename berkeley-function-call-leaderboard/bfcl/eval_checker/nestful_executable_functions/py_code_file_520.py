from typing import Tuple



def fibonacci_element(fib_seq: Tuple[int, int], n: int) -> int:

    """Computes the n-th element of the Fibonacci sequence, given a tuple `fib_seq` of the first two terms of the Fibonacci sequence.



    Args:

        fib_seq: A tuple containing the first two terms of the Fibonacci sequence.

        n: The position of the element in the Fibonacci sequence.



    Returns:

        The n-th element of the Fibonacci sequence.

    """

    if n == 1:

        return fib_seq[0]

    elif n == 2:

        return fib_seq[1]

    else:

        return fibonacci_element(fib_seq, n - 1) + fibonacci_element(fib_seq, n - 2)

