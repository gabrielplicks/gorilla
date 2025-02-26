from collections import Counter

from typing import Dict, List, Union



def count_unique_items(items: List[Union[str, int, float]]) -> Dict[Union[str, int, float], int]:

    """

    Counts the unique items in a list and returns a dictionary containing unique items and their frequencies.

    Args:

        items: A list of items to count.

    """

    assert isinstance(items, list)

    assert items

    assert all(isinstance(item, (str, int, float)) for item in items)

    counter = Counter(items)

    return dict(counter)

