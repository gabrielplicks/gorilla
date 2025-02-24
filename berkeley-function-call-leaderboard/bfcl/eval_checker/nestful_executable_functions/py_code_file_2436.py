from typing import List



def read_file_numbers(file_path: str) -> List[int]:

    """Reads a file containing numbers and returns a list of unique numbers.



    Args:

        file_path: The path to the file containing numbers.

    """

    with open(file_path, 'r') as file:

        numbers = []

        for line in file:

            line_numbers = map(int, line.split())

            numbers.extend(line_numbers)

    return list(set(numbers))

