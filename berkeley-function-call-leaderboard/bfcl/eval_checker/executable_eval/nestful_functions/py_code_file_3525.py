from typing import Dict, Union



def get_mimetype(filename: str) -> str:

    """Returns the MIME type of a file based on its extension.

    Args:

        filename: The name of the file.

    Returns:

        The MIME type of the file.

    """

    mimetypes: Dict[str, str] = {

        'jpeg': 'image/jpeg',

        'jpg': 'image/jpeg',

        'png': 'image/png',

        'pdf': 'application/pdf',

    }



    if '.' in filename:

        ext: Union[str, None] = filename.split('.')[-1].lower()

        if ext in mimetypes:

            return mimetypes[ext]

    return 'application/octet-stream'

