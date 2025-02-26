import re



def get_album_id(path: str) -> int:

    """Extracts the album ID from a file path string.



    Args:

        path: A string in the format `%album_id%-%album_name%.mp3`.



    Returns:

        The album ID as an integer, or `None` if the input is invalid.

    """

    pattern = r'^(\d+)-.*\.mp3$'  # Matches numbers followed by a hyphen and any characters until the file extension

    match = re.match(pattern, path)

    if match:

        return int(match.group(1))

    else:

        return None

