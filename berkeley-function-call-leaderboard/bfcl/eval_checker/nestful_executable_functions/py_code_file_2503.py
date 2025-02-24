from typing import Union



def evaluate_arithmetic(expression: str) -> str:

    """Evaluates an arithmetic expression and returns the result as a string, rounded to 3 decimal places.



    Args:

        expression: A string representing an arithmetic operation.



    Raises:

        Exception: If the input string does not represent a valid arithmetic operation.

    """

    try:

        result = eval(expression)

        return f'{round(result, 3):.3f}'

    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:

        raise Exception('Invalid arithmetic expression')

