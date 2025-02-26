import re

import typing



def extract_function_number(text: str) -> typing.Optional[str]:

    """Extracts the function number defined in the text using regular expressions.



    The function number is defined as a number that appears within parentheses after the word "function" in the text.

    The function returns the extracted number as a string, or None if no function number is found.



    Args:

        text: The input text to search for the function number.



    Returns:

        The extracted function number as a string, or None if no function number is found.

    """

    function_number_match = re.search(r'function\s*\((\d+)\)', text)

    if function_number_match:

        return function_number_match.group(1)

    else:

        return None

