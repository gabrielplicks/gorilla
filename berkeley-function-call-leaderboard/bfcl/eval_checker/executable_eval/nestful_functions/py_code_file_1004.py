import re



def find_all_regex_matches(file_path: str, pattern: str) -> list:

    """Finds all lines in a file containing a match for a given regular expression pattern.



    Args:

        file_path: The path to the file.

        pattern: The regular expression pattern to search for.



    Returns:

        A list of line numbers that contain matches.

    """

    line_numbers = []

    with open(file_path, 'r') as file:

        for line_number, line in enumerate(file, 1):

            if re.search(pattern, line):

                line_numbers.append(line_number)

    return line_numbers

