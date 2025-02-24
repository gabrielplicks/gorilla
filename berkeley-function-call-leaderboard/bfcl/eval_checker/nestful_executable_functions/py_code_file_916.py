from typing import Tuple



def concat_if_tuple_and_strings(collection: Tuple) -> str:

    """Concatenates the string elements of a tuple if the tuple contains only strings.



    Args:

        collection: The tuple to check and concatenate.



    Raises:

        TypeError: If the collection is not a tuple or if the tuple contains non-string elements.

    """

    if not isinstance(collection, tuple):

        raise TypeError('collection must be a tuple')



    for element in collection:

        if not isinstance(element, str):

            raise TypeError('tuple must contain only strings')



    return ''.join(collection)

