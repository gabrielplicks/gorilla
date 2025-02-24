import re

from typing import List



def extract_specific_information(input_string: str, regex_pattern: str, delimiters: str) -> List[int]:

    """Extracts specific information from a string input based on given regular expressions and delimiters.



    Args:

        input_string: The input string to extract information from.

        regex_pattern: The regular expressions to match specific patterns in the input string.

        delimiters: The delimiters to split the input string.



    Returns:

        A list of integers representing the extracted information.

    """

    matches = re.findall(regex_pattern, input_string)

    split_string = input_string.split(delimiters)

    extracted_information = [int(match) for match in matches]

    return extracted_information

