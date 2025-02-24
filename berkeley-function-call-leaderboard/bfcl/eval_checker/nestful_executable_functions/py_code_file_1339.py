import ast

from typing import List



def extract_argument_names(funcdef_node: ast.FunctionDef) -> List[str]:

    """Extracts the type names of all arguments in a function definition node from the abstract syntax tree (AST).



    Args:

        funcdef_node: The function definition node from the AST.



    Returns:

        A list of argument type names that are strings and do not start with an underscore.

    """

    try:

        return [arg.arg for arg in funcdef_node.args.args if isinstance(arg.arg, str) and not arg.arg.startswith('_')]

    except AttributeError:

        return []

