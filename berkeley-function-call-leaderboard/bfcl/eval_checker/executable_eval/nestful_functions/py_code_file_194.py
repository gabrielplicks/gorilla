import mimetypes

from typing import Optional



def detect_file_type(file_path: str) -> Optional[str]:

    """Detects the file type of a given file path.



    Args:

        file_path: The path to the file.



    Returns:

        The file type as a string, such as `jpg`, `doc`, `zip`, and so on.

        Returns None if the file type could not be detected.



    Raises:

        FileNotFoundError: If the file does not exist.

        ValueError: If the file type could not be detected.

    """

    try:

        file_type, _ = mimetypes.guess_type(file_path)

        return file_type

    except FileNotFoundError:

        raise FileNotFoundError(f"File not found: {file_path}")

    except ValueError:

        raise ValueError(f"Could not detect file type for: {file_path}")

