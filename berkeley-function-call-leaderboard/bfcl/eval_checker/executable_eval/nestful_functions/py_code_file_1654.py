from typing import Any



def check_valid_python_expression(expression: str) -> bool:

    """Checks if a given string is a valid Python expression.



    Args:

        expression: The string to check.



    Returns:

        True if the expression is valid, False otherwise.

    """

    assert type(expression) == str



    try:

        eval(expression)

        return True

    except Exception:

        return False

