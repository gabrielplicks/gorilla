import ast

from typing import Any



def check_output(snippet: str, expected_output: str) -> bool:

    """Checks if the output of a Python code snippet matches the expected output.



    Args:

        snippet: A string containing a Python code snippet.

        expected_output: A string representing the expected output of the snippet.



    Returns:

        True if the output of the snippet matches the expected output, and False otherwise.

    """

    namespace = {}

    exec(snippet, namespace)

    actual_output = namespace['snippet']()

    actual_output = ''.join(ast.literal_eval(repr(actual_output)))

    expected_output = ''.join(ast.literal_eval(repr(expected_output)))

    actual_output = ast.literal_eval(repr(actual_output))

    expected_output = ast.literal_eval(repr(expected_output))

    return actual_output == expected_output

