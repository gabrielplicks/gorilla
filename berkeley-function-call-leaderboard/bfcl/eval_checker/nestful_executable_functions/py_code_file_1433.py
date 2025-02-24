from typing import List



def extract_code_points(string: str) -> List[int]:

    """Extracts the code point values for each character in a Unicode string.



    Args:

        string: The input Unicode string.



    Returns:

        A list of code point values for each character in the input string.

    """

    return [ord(char) for char in string]

