from typing import AnyStr



def replace_double_curly_braces(input_str: AnyStr) -> AnyStr:

    """Replaces all occurrences of `{{` or `}}` with a single curly brace `{` or `}` respectively.



    Args:

        input_str: The input string to be processed.



    Returns:

        The processed string with all occurrences of `{{` or `}}` replaced with a single curly brace `{` or `}`.

    """

    return input_str.replace("{{", "{").replace("}}", "}")

