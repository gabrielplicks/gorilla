from typing import Tuple



def count_char_occurrences(filename: str, char: str) -> int:

    """Counts the occurrences of a specific character in a given text file.

    Args:

        filename: The name of the file to read.

        char: The character to count.

    """

    total_count = 0

    chunk_size = 1024 * 1024  # 1 MB chunk size

    with open(filename, 'r') as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:

                break

            chunk = chunk.replace('\n', '').replace('\t', '')

            total_count += chunk.count(char)

    return total_count

