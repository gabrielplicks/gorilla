from typing import AnyStr



def unescape_raw_string_literal(literal: AnyStr) -> AnyStr:

    """Unescapes a string literal that was previously escaped using Python's "raw" string literal encoding (prefixed with "r").



    Args:

        literal: The raw string literal to be unescaped.



    Returns:

        The unescaped string.

    """

    if literal.startswith("r"):

        literal = literal[1:]  # Remove the "r" prefix

    literal = literal.replace(r"\n", "\n")  # Unescape the "\n" escape sequence

    literal = literal.replace(r"\\", "\\")  # Unescape the "\\" escape sequence

    return literal

