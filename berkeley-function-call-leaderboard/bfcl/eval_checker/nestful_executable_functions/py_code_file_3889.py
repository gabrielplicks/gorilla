from typing import TextIO



def replace_substring_in_file(file_path: str, old_text: str, new_text: str) -> int:

    """Replaces a specific substring in all lines of a text file.

    Args:

        file_path: The path to the text file.

        old_text: The substring to be replaced.

        new_text: The substring to replace with.

    Returns:

        The number of replacements made.

    """

    with open(file_path, 'r+') as file:

        lines = file.readlines()

        num_replacements = 0

        for idx, line in enumerate(lines):

            new_line = line.replace(old_text, new_text)

            if new_line != line:

                num_replacements += 1

            lines[idx] = new_line

        file.seek(0)

        file.writelines(lines)

        file.truncate()

    return num_replacements

