from functools import reduce

from typing import List



def summarize(data: List[str]) -> List[int]:

    """Calculates the summary of the given data list.



    The summary includes the sum of all elements of the data list, as well as the number of elements in the data list.



    Args:

        data: A list of data elements.



    Returns:

        A list containing the sum and the number of elements in the data list.

    """

    mapped_data = list(map(int, data))

    sum_data = reduce(lambda x, y: x + y, mapped_data)

    summary = [sum_data, len(data)]

    return summary

