from typing import List



def ascii_to_text(ascii_codes: List[int]) -> str:

    """Converts a list of ascii executable_functions to the corresponding text.



    Args:

        ascii_codes: A list of ascii executable_functions.



    Returns:

        The corresponding text.

    """

    return ''.join(chr(code) for code in ascii_codes)

