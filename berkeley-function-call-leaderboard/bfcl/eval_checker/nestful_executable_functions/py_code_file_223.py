from typing import AnyStr



def indent_code(code: AnyStr, indentation_level: int) -> AnyStr:

    """

    Indents the input code string by the specified number of spaces.



    Args:

        code: The input code string.

        indentation_level: The desired level of indentation.



    Returns:

        A string with the code indented by the specified number of spaces.

    """

    lines = code.splitlines()

    indented_lines = [indentation_level * " " + line for line in lines]

    return "\n".join(indented_lines)

