from typing import Union



def image_filename(index: Union[int, str], prefix: str = 'image', suffix: str = 'jpg') -> str:

    """Constructs a filename string with the given prefix and suffix.



    Args:

        index: The numeric index to be included in the filename.

        prefix: The prefix to be used in the filename. Defaults to 'image'.

        suffix: The suffix to be used in the filename. Defaults to 'jpg'.

    """

    if isinstance(index, int) and index >= 0:

        filename = f"{prefix}_{index}.{suffix}"

    else:

        filename = f"{prefix}_{index}.jpg"

    return filename

