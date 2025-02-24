from typing import Optional



def wrap_print(input_string: str) -> Optional[str]:

    """Wraps a string in a print() call based on the number of newline characters.



    Args:

        input_string: The input string to be wrapped in a print() call.



    Returns:

        The input string wrapped in a print() call, indented if necessary.

    """

    if not input_string:

        return ""

    elif input_string.count("\n") > 1:

        return f"    print({input_string!r})"

    else:

        return f"print({input_string!r})"

