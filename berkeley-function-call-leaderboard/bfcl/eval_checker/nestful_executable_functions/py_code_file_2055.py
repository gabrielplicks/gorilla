from typing import Any



class FileHandler:

    def __init__(self, filename: str):

        self.filename = filename



    def close(self):

        print("Closing file...")



def test_file_handler(obj: Any) -> bool:

    """Checks if the object is an instance of FileHandler and the filename attribute is not empty.



    Args:

        obj: The object to be checked.



    Returns:

        True if the object is an instance of FileHandler and the filename attribute is not empty, False otherwise.

    """

    if isinstance(obj, FileHandler) and obj.filename:

        return True

    else:

        return False

