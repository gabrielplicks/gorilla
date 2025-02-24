import re

from typing import Tuple



def extract_file_ext(filepath: str) -> Tuple[str, bool]:

    """Extracts the file extension from a filepath and checks if it's a valid Python extension.



    Args:

        filepath: The filepath to extract the extension from.



    Returns:

        A tuple containing the file extension and a boolean indicating whether the extension is a valid Python extension.

    """

    if not isinstance(filepath, str) or not filepath:

        return None, False



    match = re.search(r'\.(\w+)$', filepath)

    if match and match.group(1) in ['py', 'pyc']:

        return match.group(0), True



    return None, False

