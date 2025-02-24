import filecmp



def are_same_files(file1: str, file2: str) -> bool:

    """Checks if two files have the same content.



    Args:

        file1: The path to the first file.

        file2: The path to the second file.



    Returns:

        True if the files have the same content, False otherwise.

    """

    return filecmp.cmp(file1, file2)

