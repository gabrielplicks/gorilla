from typing import Any



def compare_expression(p: str, q: str) -> bool:

    """Compares the output of a multi-line expression to a string.



    Args:

        p: A multi-line expression to evaluate.

        q: A string to compare to the result of the evaluation of `p`.



    Returns:

        True if the two strings are equal, False otherwise.

    """

    p = p.replace("\n", "")

    p = p.replace(" ", "")

    result = eval(p)

    return str(result) == q

