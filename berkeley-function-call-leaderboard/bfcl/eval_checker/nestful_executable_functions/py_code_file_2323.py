import os

import typing



def generate_file_names(base_path: str, names: typing.List[str]) -> typing.List[str]:

    """

    Generates file names for a given list of strings.

    Args:

        base_path: The path to a base directory.

        names: A list of strings.

    """

    file_names = []



    for name in names:

        file_names.append(os.path.join(base_path, name))



    return file_names

