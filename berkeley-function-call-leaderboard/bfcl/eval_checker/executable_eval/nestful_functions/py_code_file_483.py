import ast



def generate_ast(expression: str) -> ast.AST:

    """Generates a Python AST to evaluate a given mathematical expression.



    Args:

        expression: A string representing a mathematical expression.



    Returns:

        An AST representing the parsed expression.

    """

    expr = ast.parse(expression, mode='eval')

    return expr

