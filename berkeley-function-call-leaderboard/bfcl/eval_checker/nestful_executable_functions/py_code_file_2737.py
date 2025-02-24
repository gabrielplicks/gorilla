import os



def expand_tilde(path: str) -> str:

    """

    Expands the tilde "~" in a file path to the current user's home directory.

    Returns the expanded path if the path contains a tilde, otherwise returns the path as-is.

    Args:

        path: The file path to be expanded.

    """

    if path.startswith('~'):

        expanded_path = os.path.expanduser(path)

        if expanded_path != path:

            return expanded_path

    if path.startswith('~/'):

        home_dir = os.environ.get('HOME')

        current_dir = os.environ.get('PWD')

        if home_dir is not None and current_dir is not None:

            return os.path.join(home_dir, path[2:])

    return path

