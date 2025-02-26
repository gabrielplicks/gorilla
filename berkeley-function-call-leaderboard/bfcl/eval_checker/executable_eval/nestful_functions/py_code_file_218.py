from typing import Tuple



def check_file_pattern(file_path: str, pattern: Tuple[bytes, int], n: int) -> bool:

    """Checks if the first n bytes of a file match a specific pattern.

    Args:

        file_path: The path to the file to be checked.

        pattern: A 2-element tuple containing the expected first byte and the number of bytes to read.

        n: The number of bytes to read from the file.

    """

    with open(file_path, 'rb') as f:

        file_start = f.read(n)

        return file_start.startswith(pattern)

