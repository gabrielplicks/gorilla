from typing import BinaryIO



def compare_files(file1_path: str, file2_path: str) -> bool:

    """Compares the contents of two files byte-by-byte.



    Args:

        file1_path: The path to the first file.

        file2_path: The path to the second file.



    Returns:

        True if the files are equal, False otherwise.

    """

    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:

        file1_contents = file1.read()

        file2_contents = file2.read()



        if len(file1_contents) != len(file2_contents):

            return False



        for byte1, byte2 in zip(file1_contents, file2_contents):

            if byte1 != byte2:

                return False



    return True

