from typing import Tuple



def get_output_file(file_name: str, flag: bool) -> str:

    """

    Returns the name of the output file based on the input file name and flag.

    If the flag is True, the output file name will have the suffix "_reversed.txt".

    If the flag is False, the output file name will have the suffix "_sorted.txt".



    Args:

        file_name: The name of the input file.

        flag: A boolean value indicating whether the output file should be reversed or sorted.



    Returns:

        The name of the output file.

    """

    base_name, file_extension = file_name.rsplit('.', 1)

    if flag:

        output_file = f'{base_name}_reversed.{file_extension}'

    else:

        output_file = f'{base_name}_sorted.{file_extension}'

    return output_file

