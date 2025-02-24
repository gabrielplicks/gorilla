from typing import List



def percentages(elements: List[float]) -> List[float]:

    """Transforms each element in a list into a percentage representation.



    Each element is divided by the sum of all elements in the list, then multiplied by 100.



    Args:

        elements: A list of elements to be transformed into percentages.



    Returns:

        A list of percentage representations of each element.

    """

    total = sum(elements)

    return [100 * e / total for e in elements]

