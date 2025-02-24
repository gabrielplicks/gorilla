from datetime import datetime

from typing import Union



def get_file_timestamp(filename: Union[str, bytes]) -> str:

    """Retrieves the current datetime as a string and appends it to the given filename.



    Args:

        filename: A string or bytes representing the file path to be appended with a timestamp.



    Returns:

        A string representing the file path with a timestamp appended.

    """

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    return f'{filename}_{timestamp}'

