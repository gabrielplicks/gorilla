import ast

from typing import Any



def is_string_literal(node: Any) -> bool:

    """Checks if an AST node represents a string literal.

    Args:

        node: The AST node to be checked.

    Returns:

        A boolean value indicating whether the node represents a string literal.

    """

    if isinstance(node, ast.Str):

        return True

    return False

