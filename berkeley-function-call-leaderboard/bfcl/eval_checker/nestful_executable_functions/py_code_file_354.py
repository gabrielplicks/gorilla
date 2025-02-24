from typing import Tuple



def check_file_permissions(a: int, b: int) -> bool:

    """Determines if all of the file permissions in `a` are also present in `b`, using bitwise operations.



    Args:

        a: A combination of bitwise OR of some file permissions.

        b: A combination of some other file permissions.



    Returns:

        True if all of the file permissions in `a` are also present in `b`, False otherwise.

    """

    return (a & b) == a and (a ^ b) == 0

