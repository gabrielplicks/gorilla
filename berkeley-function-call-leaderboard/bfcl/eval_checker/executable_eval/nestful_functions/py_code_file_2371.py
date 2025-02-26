from typing import Any, Union



def sum_if_int(collection: list[Any]) -> float:

    """Calculates the sum of all integers and floats in a collection.

    Args:

        collection: A list of any type of elements.

    """

    total = 0

    for item in collection:

        if isinstance(item, (int, float)):

            total += item



    return total

