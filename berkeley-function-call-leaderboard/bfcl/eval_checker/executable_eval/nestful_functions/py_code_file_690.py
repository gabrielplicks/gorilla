import ast



def check_while_statements(source_code: str) -> bool:

    """Checks if a Python function only accepts `while` statements.



    Args:

        source_code: The source code of the function.



    Returns:

        True if the function body only consists of `while` statements, False otherwise.

    """

    tree = ast.parse(source_code)

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):

            for statement in node.body:

                if not isinstance(statement, ast.While):

                    return False

    return True

