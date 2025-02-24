import re

import typing



def extract_alphanumeric(input_string: str) -> str:

    """Extracts alphanumeric characters from a given string.



    Args:

        input_string: The input string to extract alphanumeric characters from.



    Returns:

        A string containing only the alphanumeric characters from the original string.

    """

    try:

        alphanumeric_string = re.sub(r'\W+', '', input_string)  # Replace non-alphanumeric characters with empty string

        return alphanumeric_string

    except Exception as e:

        print(f'Error: {str(e)}')

        return ''

