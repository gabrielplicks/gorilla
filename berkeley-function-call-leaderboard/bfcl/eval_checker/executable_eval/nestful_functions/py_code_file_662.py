from typing import List, Union



def count_empty_strings(strings: List[Union[str, None]]) -> int:

    """Counts the number of empty strings in a collection of strings.



    Args:

        strings: A collection of strings.



    Returns:

        The number of empty strings in the collection.

    """

    count = 0

    for string in strings:

        if string is None:

            continue

        if string == '':

            count += 1

    return count

