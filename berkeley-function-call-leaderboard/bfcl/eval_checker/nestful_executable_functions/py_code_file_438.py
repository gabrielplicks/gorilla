import gzip

from typing import TextIO



def count_lines_gzip(filename: str) -> int:

    """Counts the number of non-empty lines in a .gz file.



    Args:

        filename: The name of the .gz file.



    Returns:

        The number of non-empty lines in the file.

    """

    num_lines = 0



    with gzip.open(filename, 'rt') as f:

        for line in f:

            line = line.strip()

            if line:

                num_lines += 1



    return num_lines

