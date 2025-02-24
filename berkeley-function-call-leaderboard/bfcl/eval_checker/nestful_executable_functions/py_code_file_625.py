import re

from typing import Union



def check_match(pattern: str, file_name: Union[str, bytes]) -> bool:

    """Checks if a regular expression pattern matches a file name.



    Args:

        pattern: The regular expression pattern to match against the file name.

        file_name: The name of the file to check for a match.



    Returns:

        True if the pattern matches the file name, False otherwise.

    """

    try:

        with open(file_name) as f:

            file_prefix = f.name.split('.')[0]

            return bool(re.match(pattern, file_prefix))

    except IOError as e:

        print(f"Error: {e}")

        return False

