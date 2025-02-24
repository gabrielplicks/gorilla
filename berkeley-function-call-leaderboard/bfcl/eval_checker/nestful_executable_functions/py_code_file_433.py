from typing import List



def first_item(collection1: List, collection2: List) -> int:

    """Returns the first item from the longer collection.

    If the two collections are of equal length, returns the first item from the first collection.

    Args:

        collection1: The first collection.

        collection2: The second collection.

    """

    if len(collection1) > len(collection2):

        return collection1[0]

    elif len(collection2) > len(collection1):

        return collection2[0]

    else:

        return collection1[0]

