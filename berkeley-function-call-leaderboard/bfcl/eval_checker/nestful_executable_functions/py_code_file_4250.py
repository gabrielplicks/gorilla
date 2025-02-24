from typing import Tuple



def create_data_file_listing_command(dir: str, file: str) -> str:

    """

    Generates a bash command to list the contents of a directory and output the results to a file.

    The command should conform to the syntax: `ls -l <dir> > file.txt`.

    Args:

        dir: The directory to list the contents of.

        file: The file to output the results to.

    """

    return f"ls -l {dir} > {file}"

