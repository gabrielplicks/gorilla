from typing import List



def replace_suffix(paths: List[str], suffix: str) -> List[str]:

    """Replaces the suffix of a list of filepaths with a new suffix.



    Args:

        paths: A list of filepaths.

        suffix: The new suffix to replace the existing suffix with.



    Returns:

        A new list of filepaths with the specified suffix replaced.

    """

    new_paths = []

    for path in paths:

        if path.endswith(suffix):

            new_paths.append(path)

        else:

            new_paths.append(path.replace(path.split(".")[-1], suffix))

    return new_paths

