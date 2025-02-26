import re

from typing import List



def extract_floats(text: str) -> List[float]:

    """Extracts all floating point numbers from a given text string.



    Args:

        text: The input text string.



    Returns:

        A list of floating point numbers found in the text.

    """

    float_pattern = r'\d+(?:\.\d+)?'

    matches = re.findall(float_pattern, text)

    return [float(match) for match in matches if is_float_safe(match)]



def is_float_safe(number: str) -> bool:

    """Checks if a number can be safely parsed as a float.



    Args:

        number: The input number as a string.



    Returns:

        True if the number can be safely parsed as a float, False otherwise.

    """

    try:

        float(number)

        return True

    except ValueError:

        return False

