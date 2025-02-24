from typing import Tuple



def create_new_path(existing_path: str, new_last_part: str) -> str:

    """Creates a new path by replacing the last part of an existing path.

    If the existing path ends with a slash, the new path should also end with a slash.

    Args:

        existing_path: The existing path.

        new_last_part: The new last part of the path.

    Returns:

        The new path.

    """

    path_parts: Tuple[str, str] = existing_path.rsplit('/', maxsplit=1)

    new_path: str = '/'.join([path_parts[0], new_last_part])

    return new_path

