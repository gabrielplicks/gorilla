import re

from typing import AnyStr



pattern = re.compile(r"abc")

replacement = "MACRO_X"



def replace_abc_with_macro(input_string: AnyStr) -> AnyStr:

    """Replaces all occurrences of the literal string "abc" in the input string with the macro name MACRO_X.



    Args:

        input_string: The input string to search and replace in.



    Returns:

        The input string with all occurrences of "abc" replaced with MACRO_X.

    """

    return pattern.sub(replacement, input_string)

