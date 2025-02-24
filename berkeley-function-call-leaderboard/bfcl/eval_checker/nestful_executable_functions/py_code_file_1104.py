import fnmatch



def compare_file_paths(path1: str, path2: str) -> bool:

    """

    Compares two file paths based on wildcard matching rules.



    Args:

        path1: The first file path.

        path2: The second file path.



    Returns:

        True if the paths match according to the rules, False otherwise.

    """

    if '*' in path1 or '?' in path1:

        return fnmatch.fnmatch(path2, path1)

    elif '*' in path2 or '?' in path2:

        return fnmatch.fnmatch(path1, path2)

    else:

        return len(path1) == len(path2) and all(c1 == c2 for c1, c2 in zip(path1, path2))

