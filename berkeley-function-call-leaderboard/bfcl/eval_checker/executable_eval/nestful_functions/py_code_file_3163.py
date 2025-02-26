import re

from typing import List, Tuple



def search_all_matches(pattern: str, file_path: str) -> List[Tuple[int, str]]:

    """Searches for all occurrences of a pattern in a file and returns the results as a list of tuples.

    Each tuple represents a match and contains the line number and the matching text.

    The pattern can be a simple string or a regular expression.

    Args:

        pattern: The pattern to search for.

        file_path: The path to the file to search in.

    """

    results = []

    line_number = 1



    with open(file_path, 'r') as file:

        for line in file:

            matches = re.findall(pattern, line)

            for match in matches:

                results.append((line_number, match))

            line_number += 1



    return results

