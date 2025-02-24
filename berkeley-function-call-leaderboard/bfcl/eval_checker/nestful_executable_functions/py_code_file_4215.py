from typing import Union



def generate_filepath_for_number(num: Union[int, float], dir_path: str) -> str:

    """Generates a file path based on the provided number and directory path.

    The file path is formatted as: `dir_path/n/num.txt`, where `n` represents the last digit of `num`.

    If `num` is less than `10`, prepend a zero to the filename.



    Args:

        num: The number used to generate the file path.

        dir_path: The directory path where the file will be located.



    Returns:

        The generated file path.

    """

    n = num % 10

    file_path = f"{dir_path}/{n}/{num:04}.txt"



    return file_path

