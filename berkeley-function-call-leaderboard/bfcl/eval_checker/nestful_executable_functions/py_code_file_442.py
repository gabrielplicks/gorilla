from typing import Any



def eval_condition(statement: str) -> str:

    """Evaluates a conditional statement in the global scope of the current script.

    Args:

        statement: The conditional statement to evaluate.

    Returns:

        The string "True" if the statement evaluates to True, and "False" otherwise.

    """

    if eval(statement, globals()):

        return "True"

    else:

        return "False"

