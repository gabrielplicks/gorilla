from typing import Tuple



def backward_step_recursion(n: int, prev: int) -> Tuple[int, int]:

    """Computes the backward step of the recursive formula: f(n) = n * f(n - 1), where n is a positive integer and f(0) = 1.



    Args:

        n: A positive integer.

        prev: The result of the previous step.



    Returns:

        A tuple containing two elements: the first element is the current step result, and the second element is the next step's `prev` value. If `n` is not positive, the function returns `None`.

    """

    if n <= 0:

        return None



    current_result = n * prev

    next_prev = current_result



    return (current_result, next_prev)

