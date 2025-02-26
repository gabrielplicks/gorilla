from typing import List



def check_unique_lines(filename: str) -> bool:

    """Checks if all the lines in a file are unique.



    A single line is considered unique if it is identical to another line,

    but not if it is only different in case. For instance, the lines "Hello"

    and "hello" are considered identical in this function.



    Args:

        filename: The name of the file to check.



    Returns:

        A boolean indicating whether all the lines in the file are unique.

    """

    with open(filename, 'r') as f:

        lines: List[str] = [line.strip().lower() for line in f]

    num_lines: int = len(lines)

    num_unique_lines: int = len(set(lines))

    return num_unique_lines == num_lines

