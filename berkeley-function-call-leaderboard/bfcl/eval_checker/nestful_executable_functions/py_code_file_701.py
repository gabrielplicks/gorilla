import os



def append_path_components(path_components: list[str]) -> str:

    """

    Appends a list of path components into a single path string.

    The function handles platform-specific path separators appropriately.

    It does not remove any leading or trailing separators.



    Args:

        path_components: A list of path components.

    """

    return os.path.join(*[component for component in path_components if component])

