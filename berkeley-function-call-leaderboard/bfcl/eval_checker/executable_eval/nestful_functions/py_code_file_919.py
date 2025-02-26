from typing import Union



def is_audio_file(filename: str) -> bool:

    """Checks if a file name ends with `.wav` or `.aiff`.



    Args:

        filename: The name of the file.



    Returns:

        True if the file name ends with `.wav` or `.aiff`, False otherwise.

    """

    return filename.endswith('.wav') or filename.endswith('.aiff')

