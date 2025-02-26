from typing import Optional



def check_path_exists(path_a: Optional[str], path_b: str, path_c: str) -> bool:

    """Checks if path_a exists and does not match any of path_b or path_c.



    Args:

        path_a: The path to check.

        path_b: The first path to compare with path_a.

        path_c: The second path to compare with path_a.



    Returns:

        False if path_a exists and does not match any of path_b or path_c, and True otherwise.

    """

    if path_a is None:

        return False

    if not path_a:

        return False

    if path_a == path_b or path_a == path_c:

        return False



    return True

