from typing import Union



def simplify_conditional_expression(exp: str) -> Union[str, bool]:

    """Simplifies a conditional expression.



    Args:

        exp: A string consisting of one or more conditional operations (a ? b : c)

            with the following format:

            - a is a boolean expression

            - b is a boolean expression

            - c is a boolean expression



    Returns:

        The simplified expression.

    """

    tokens = exp.split()

    if tokens[0] == "true":

        return tokens[2]

    else:

        return tokens[4]

