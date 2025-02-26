from typing import List



def replace_and_sort_file_content(filename: str) -> None:

    """

    Reads a file, replaces all occurrences of the word "old" with the word "new",

    removes all duplicate lines, sorts the remaining lines, and saves the updated

    content to a new file with the same name and the suffix "-v2".



    Args:

        filename: The name of the file to read and update.

    """

    with open(filename) as f:

        lines: List[str] = f.readlines()

    lines = [line.replace("old", "new") for line in lines]

    unique_lines: List[str] = sorted(set(lines))

    new_filename: str = f"{filename}-v2"

    with open(new_filename, "w") as f:

        f.writelines(unique_lines)

