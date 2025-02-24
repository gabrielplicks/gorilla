import atexit

import tempfile



def create_temp_dir_and_write(content: str) -> str:

    """

    Creates a temporary directory and writes the given `content` to it.

    The temporary directory is deleted after the program exits.

    Args:

        content: The content to be written to the temporary file.

    """

    temp_dir = tempfile.TemporaryDirectory()

    atexit.register(temp_dir.cleanup)

    with open(f"{temp_dir.name}/temp_file.txt", "w") as f:

        f.write(content)



    return temp_dir.name

