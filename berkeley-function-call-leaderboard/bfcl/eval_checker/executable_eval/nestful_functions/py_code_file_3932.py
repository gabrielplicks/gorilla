import os

from typing import List



def concat_files(file_list: List[str]) -> None:

    """Concatenates the contents of the given files into a new file called 'out.txt'.



    Args:

        file_list: A list of file names.



    Raises:

        FileNotFoundError: If any of the files do not exist.

    """

    output_file = 'out.txt'

    try:

        with open(output_file, 'w') as out_f:

            for file_name in file_list:

                if not os.path.isfile(file_name):

                    raise FileNotFoundError(f"File '{file_name}' not found")

                with open(file_name, 'r') as in_f:

                    out_f.write(in_f.read())

    except FileNotFoundError as e:

        print(f"Error: {e}")

