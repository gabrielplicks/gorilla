from typing import Tuple



def compute_nth_fibonacci(n: int) -> Tuple[int, bool]:

    """Computes the nth term of the Fibonacci sequence.



    Args:

        n: The term to compute.



    Returns:

        A tuple containing the nth term and a boolean to indicate whether it is valid.

        If the input is not a non-negative integer, the function returns (None, False).

    """

    if not isinstance(n, int):

        return (None, False)

    if n < 0:

        return (None, False)



    fibonacci = [0, 1]

    for i in range(2, n+1):

        fibonacci.append(fibonacci[i-1] + fibonacci[i-2])



    return (fibonacci[-1], True)

