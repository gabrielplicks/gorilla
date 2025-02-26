from typing import List



def read_and_truncate_lines(file_path: str, max_length: int = 64) -> List[str]:

    """Reads a text file and splits it into lines, where each line is a separate string in a list.

    Additionally, the function supports an additional argument `max_length` (default is 64 characters per line)

    to truncate long lines. For each line, truncate it to the specified length while omitting any trailing whitespace.



    Args:

        file_path: The path to the text file.

        max_length: The maximum length of each line (default is 64 characters).



    Returns:

        A list of strings, where each string is a line from the text file truncated to the specified length.

    """

    with open(file_path, "r") as f:

        lines = [line.rstrip()[:max_length] for line in f.readlines()]

    return lines

