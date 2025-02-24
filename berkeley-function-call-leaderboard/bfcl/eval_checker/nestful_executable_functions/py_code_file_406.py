import os

import re

from typing import Optional



def get_timestamp_from_filename(filepath: str) -> Optional[str]:

    """Extracts a timestamp in the format `YYYYMMDDHHMMSS` from the filename of an image file.



    Args:

        filepath: The file path of the image file.



    Returns:

        The timestamp as a string, or None if no timestamp is found in the filename.

    """

    filename = os.path.basename(filepath)

    match = re.search(r"\d{14}", filename)

    if match:

        return match.group()

    else:

        return None

