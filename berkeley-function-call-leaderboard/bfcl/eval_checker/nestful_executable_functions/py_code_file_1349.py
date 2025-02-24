import ast

from typing import Union



def parse_code_snippet(code_snippet: str) -> Union[ast.AST, None]:

    """Parses a Python code snippet and returns a normalized `ast` representation.



    Args:

        code_snippet: A string containing the Python code snippet.



    Returns:

        An `ast.AST` object representing the normalized code snippet, or `None` if the code snippet has invalid syntax.

    """

    normalized_code_snippet = code_snippet.replace('\n', '')  # Normalize code snippet



    try:

        tree = ast.parse(normalized_code_snippet, mode='eval')

    except SyntaxError:

        print('Invalid syntax in code snippet.')

        return None



    return tree

