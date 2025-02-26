from typing import List



def compare_strings_from_file(file_path: str, string_1: str, string_2: str) -> List[int]:

    """

    Compares two strings case-insensitively in a file and returns the line numbers at which they match.

    Args:

        file_path: The path to the file.

        string_1: The first string to compare.

        string_2: The second string to compare.

    """

    line_numbers = []



    try:

        with open(file_path, 'r') as file:

            for line_number, line in enumerate(file, start=1):

                if string_1.lower() in line.lower() and string_2.lower() in line.lower():

                    line_numbers.append(line_number)

    except IOError:

        print("File does not exist or cannot be opened for reading.")



    return line_numbers

