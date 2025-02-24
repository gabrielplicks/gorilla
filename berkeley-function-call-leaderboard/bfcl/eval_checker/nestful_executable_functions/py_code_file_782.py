import re

from typing import List, Dict



def find_patterns(regex: str, input_string: str) -> List[Dict[str, int | str]]:

    """Finds all the matched patterns in the input string based on the given regular expression.



    Args:

        regex: The regular expression to search for in the input string.

        input_string: The input string to search in.



    Returns:

        A list of dictionaries, where each dictionary contains the start index, length, and matched string of a single match.

    """

    matches: List[Dict[str, int | str]] = []

    for match in re.finditer(regex, input_string):

        matches.append({

            'start_index': match.start(),

            'length': match.end() - match.start(),

            'matched_string': match.group()

        })

    return matches

