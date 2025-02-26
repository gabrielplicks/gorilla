from typing import List



def extend_range(start: float, end: float, step: int) -> List[float]:

    """Extends a range given by [start, end) by step times, where the extension is symmetric on either side of the range.

    Returns a list of numbers generated from the extended range, with each number being converted to a ratio between 0 and 1.



    Args:

        start: The start of the range.

        end: The end of the range.

        step: The number of times to extend the range.

    """

    extended_range = [start]

    for i in range(1, step + 1):

        extended_range.append(start + i * (end - start))

    extended_range.append(end)



    total_elements = len(extended_range)

    return [i / total_elements for i in range(total_elements)]

