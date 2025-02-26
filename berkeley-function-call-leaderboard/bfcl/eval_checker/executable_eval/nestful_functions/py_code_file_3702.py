from typing import List, Tuple



def find_substring_positions(string: str, substring: str) -> List[Tuple[int, int]]:

    """

    Converts a code snippet into a list of line numbers and column numbers where the given string appears.

    Each element of the returned list is a tuple consisting of the line number and column number within the string

    where the given substring is found.



    Args:

        string: The code snippet to search.

        substring: The substring to find within the code snippet.



    Returns:

        A list of tuples, where each tuple contains the line number and column number where the substring is found.

    """

    result = []

    line = 1

    column = 1



    for char in string:

        if char == '\n':

            line += 1

            column = 1

        elif char == substring:

            result.append((line, column))

        else:

            column += 1



    return result

