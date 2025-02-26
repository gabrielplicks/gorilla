import re



def read_file_version(file_path: str) -> tuple:

    """Reads the version number from a file.



    Args:

        file_path: The path to the file.



    Returns:

        A tuple containing the major, minor, and patch version numbers.

    """

    try:

        with open(file_path, 'r') as file:

            content = file.read()

            match = re.search(r"v(\d+)\.(\d+)(?:\.(.*))?", content)

            if match:

                major, minor, patch = match.groups()

                return (int(major), int(minor), patch)

    except FileNotFoundError:

        pass

    return (0, 0, '')

