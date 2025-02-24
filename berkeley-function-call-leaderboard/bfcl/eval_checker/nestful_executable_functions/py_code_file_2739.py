from typing import List, Tuple



def parse_ignore_file(ignore_file_path: str) -> List[Tuple[int, int]]:

    """Parses an ignore file and returns a list of tuples, each tuple representing a line of the ignore file.

    Each tuple contains two elements, the first being the character index, and the second being the length of the text.

    Each "character" in the text is parsed from the ignore file.



    Args:

        ignore_file_path: The path to the ignore file.

    """

    with open(ignore_file_path, "r") as ignore_file:

        ignore_lines = ignore_file.read().splitlines()



    ignore_tuples = []

    for index, line in enumerate(ignore_lines):

        ignore_tuples.append((index, len(line)))



    return ignore_tuples

