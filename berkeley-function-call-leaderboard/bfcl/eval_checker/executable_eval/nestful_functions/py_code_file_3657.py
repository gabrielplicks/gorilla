from typing import List



def to_relative(paths: List[str], root: str) -> List[str]:

    """Transforms a list of absolute file paths to relative file paths by removing the root.



    Args:

        paths: A list of absolute file paths.

        root: The root directory to remove from the paths.



    Returns:

        A list of relative file paths.

    """

    return [path.replace(root, '', 1).lstrip('\\/') for path in paths]

