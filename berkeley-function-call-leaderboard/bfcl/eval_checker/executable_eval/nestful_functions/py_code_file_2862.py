from typing import List

from tkinter import Entry



def get_text_list(objects: List[Entry]) -> List[str]:

    """Returns a list of text content from a list of objects that have a `get` method.



    Args:

        objects: A list of objects that have a `get` method.



    Returns:

        A list of text content from the objects.

    """

    return [obj.get() for obj in objects]

