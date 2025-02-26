import os

import typing



def make_relative(paths: typing.List[str]) -> typing.List[str]:

    """

    Takes a list of strings representing paths to files, and returns a list of strings with the same paths relative to the current working directory.

    The function handles paths with and without trailing slashes.



    Args:

        paths: A list of strings representing paths to files.



    Returns:

        A list of strings with the same paths relative to the current working directory.

    """

    cwd = os.getcwd()

    relative_paths = []

    for path in paths:

        if path.endswith('/'):

            path = path[:-1]

        relative_paths.append(os.path.relpath(path, cwd))

    return relative_paths

