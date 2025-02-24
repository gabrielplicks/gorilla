from typing import Union



def evaluate_instruction(a: int, b: int) -> Union[int, float]:

    """

    Evaluates the result of the given instruction.



    If `b` is even, the function returns the sum of `a` and `b`.

    If `b` is odd, the function returns the product of `a` and `b`.



    Args:

        a: The first integer.

        b: The second integer.



    Returns:

        The result of the evaluation.

    """

    return a + b if b % 2 == 0 else a * b

