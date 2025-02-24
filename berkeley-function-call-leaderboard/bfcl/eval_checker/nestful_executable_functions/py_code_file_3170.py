from typing import AnyStr

import textwrap



def format_code(code: AnyStr) -> AnyStr:

    """Formats a string of Python code by indenting each line by 4 spaces.



    The formatted string is surrounded by triple quotes, and the contents of the string are escaped to ensure that it is a valid Python string literal.



    Args:

        code: The string of Python code to format.



    Returns:

        The formatted string.

    """

    dedented_code = textwrap.dedent(code)

    indented_code = textwrap.indent(dedented_code, "    ")

    return f'"""\n{indented_code}\n"""'

