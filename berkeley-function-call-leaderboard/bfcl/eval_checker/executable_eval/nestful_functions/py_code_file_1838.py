from typing import List



def escape_reserved_keywords(string: str) -> str:

    """Escapes any reserved Python keywords in a string so that they can be used as variable names.



    Args:

        string: The input string.



    Returns:

        The input string with any reserved keywords escaped.

    """

    reserved_keywords: List[str] = ['False', 'class', 'finally', 'is', 'return', 'None', 'continue', 'for', 'lambda', 'try', 'True', 'def', 'from', 'nonlocal', 'while', 'and', 'del', 'global', 'not', 'with', 'as', 'elif', 'if', 'or', 'yield', 'assert', 'else', 'import', 'pass', 'break', 'except', 'in', 'raise']



    if string in reserved_keywords:

        return string + '_'

    else:

        return string

