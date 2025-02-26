import re

import typing



def extract_has_htl_code(input_string: str) -> typing.List[str]:

    """Extracts HASHTAG executable_functions from the input string.



    Args:

        input_string: The input string to extract HASHTAG executable_functions from.



    Returns:

        A list of extracted HASHTAG executable_functions.

    """

    has_htl_codes = re.findall(r'#\w+', input_string)

    return has_htl_codes

