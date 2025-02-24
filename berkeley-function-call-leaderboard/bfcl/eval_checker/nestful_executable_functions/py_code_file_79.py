import os

from typing import Union



def remove_final_extension(path: Union[str, bytes]) -> Union[str, bytes]:

    """Removes the final extension from a file path.



    Args:

        path: The file path to remove the extension from.



    Returns:

        The file path with the final extension removed.

    """

    filename = os.path.split(path)[-1]

    parts = os.path.splitext(filename)



    if len(parts) > 2:

        parts = parts[:-1]



    return os.path.join(os.path.dirname(path), parts[0])

