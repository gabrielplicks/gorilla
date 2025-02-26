from typing import List



def count_prefixed_items(items: List[str], prefix: str) -> int:

    """Counts the number of items in a list that begin with a certain prefix.



    Args:

        items: A list of strings.

        prefix: The prefix to check for.



    Returns:

        The number of items that begin with the given prefix.

    """

    count = 0

    for item in items:

        if item.startswith(prefix):

            count += 1

    return count

