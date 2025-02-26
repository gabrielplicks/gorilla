import os.path

from typing import Union



def expand_and_check(path: str) -> Union[str, None]:

    """Expands a path and checks if it exists.



    Args:

        path: The path to expand and check.



    Returns:

        The expanded path if it exists, otherwise `None`.

    """

    expanded_path = os.path.expanduser(path)

    if os.path.exists(expanded_path):

        return expanded_path

    else:

        return None

