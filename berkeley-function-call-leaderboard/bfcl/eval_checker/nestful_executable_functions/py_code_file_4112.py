from pathlib import Path

from typing import Union



def get_full_path_of_file(filename: Union[str, Path]) -> str:

    """Gets the full path of a file in the user's home folder.



    Args:

        filename: The name of the file.



    Raises:

        FileNotFoundError: If the file does not exist in the home folder.

    """

    home_path = Path.home()

    file_path = home_path / filename



    if not file_path.exists():

        raise FileNotFoundError(f'File not found: {file_path}')



    return str(file_path)

