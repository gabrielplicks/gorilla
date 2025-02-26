from typing import List



def read_file_and_return_numbers(file_path: str) -> List[int]:

    """Reads a file with strings on each line, and returns a list of numbers found in the file.

    Handles parsing errors gracefully, returning an empty list if an error occurs.

    Args:

        file_path: The path to the file to read.

    Returns:

        A list of integers found in the file.

    """

    numbers = []

    try:

        with open(file_path, 'r') as file:

            for line in file:

                line = line.strip()

                try:

                    number = int(line)

                except ValueError:

                    continue

                numbers.append(number)

    except IOError:

        pass

    return numbers

