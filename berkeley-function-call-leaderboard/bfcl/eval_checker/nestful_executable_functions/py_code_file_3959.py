from typing import Tuple



def path_join(path1: str, path2: str) -> str:

    """Concatenates two paths together considering the Windows convention.



    The path separator is a backslash ('\\').



    Args:

        path1: The first path.

        path2: The second path.



    Returns:

        The concatenated path.

    """

    path1 = path1.strip('\\')

    path2 = path2.strip('\\')

    if path1.endswith('\\'):

        path1 = path1[:-1]

    if path2.startswith('\\'):

        path2 = path2[1:]



    return '\\'.join([path1, path2])

