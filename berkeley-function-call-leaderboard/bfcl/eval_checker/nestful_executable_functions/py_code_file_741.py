from typing import Any



def file_checker(filepath: str) -> bool:

    """Checks if a file exists and can be opened.

    Args:

        filepath: The path to the file.

    Returns:

        True if the file exists and can be opened, False otherwise.

    """

    try:

        with open(filepath, 'r') as f:

            contents = f.read()

            return True

    except Exception:

        return False

