from typing import Callable



def tail_recur(n: int) -> int:

    """Computes the factorial of a non-negative integer using tail recursion.



    Args:

        n: The non-negative integer to compute the factorial of.



    Raises:

        ValueError: If the input number is negative.

    """

    if n < 0:

        raise ValueError("Input number must be non-negative.")

    if n == 0:

        return 1



    def helper(n: int, acc: int) -> int:

        """Helper function for tail recursion.



        Args:

            n: The current number to compute the factorial of.

            acc: The accumulator for the intermediate results.



        Returns:

            The factorial of the input number.

        """

        if n == 0:

            return acc

        return helper(n - 1, acc * n)



    return helper(n, 1)

