from typing import List, Union



def find_last_known_value(seq: List[Union[None, object]], default: Union[None, object] = None) -> Union[None, object]:

    """Finds the last non-None value in a list of hashable values.



    Args:

        seq: A list of hashable values.

        default: The value to return if no non-None value is found.

    """

    filtered = [x for x in seq if x is not None]

    for x in reversed(filtered):

        return x

    return default

