from typing import Optional



def add_suffix_to_filename(filename: Optional[str], suffix: str) -> Optional[str]:

    """Adds a suffix to a filename before the extension.



    Args:

        filename: The filename to add the suffix to.

        suffix: The suffix to add to the filename.



    Returns:

        The filename with the suffix added before the extension.

    """

    if filename is None or filename == '':

        return None



    before_dot, after_dot = filename.rsplit('.', 1) if '.' in filename else (filename, None)



    if after_dot is not None:

        return f"{before_dot}.{suffix}.{after_dot}"

    else:

        return f"{filename}.{suffix}"

