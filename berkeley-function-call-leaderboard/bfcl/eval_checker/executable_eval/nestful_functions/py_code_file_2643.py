from typing import List



def get_branches(path: str) -> List[str]:

    """Returns the parts of the path that represent branches (not including the root node).



    Args:

        path: The input path string.



    """

    path = path.rstrip('/')

    return path.split('/')[1:]

