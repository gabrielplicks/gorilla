from typing import Optional



def string_with_length(input_string: str) -> Optional[str]:

    """Returns a string that includes the input string and its length in parentheses.



    If the input string is empty, it returns the string "Invalid input".



    Args:

        input_string: The input string.



    Returns:

        A string that includes the input string and its length in parentheses, or

        the string "Invalid input" if the input string is empty.

    """

    if not input_string:

        return "Invalid input"

    return f"{input_string} ({len(input_string)})"

