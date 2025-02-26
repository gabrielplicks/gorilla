from typing import List, Dict, Tuple, Any



def count_most_frequent_elements(data: List[Any], n: int) -> List[Any]:

    """Returns the n most frequently occurring elements in the data list.

    Args:

        data: A list of arbitrary objects.

        n: A positive integer.

    """

    frequency: Dict[Any, int] = {}

    for element in data:

        frequency[element] = frequency.get(element, 0) + 1

    sorted_frequency: List[Tuple[Any, int]] = sorted(frequency.items(), key=lambda x: (-x[1], x[0]))

    return [element[0] for element in sorted_frequency[:n]]

