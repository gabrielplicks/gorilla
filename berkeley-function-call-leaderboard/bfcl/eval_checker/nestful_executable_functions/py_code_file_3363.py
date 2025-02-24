from typing import List, Tuple



def filter_array_by_timestamp(array: List[Tuple[int, str, str]], timestamp_start: int, timestamp_end: int) -> List[Tuple[int, str, str]]:

    """Filters an array of tuples by a given timestamp range (inclusive).



    Args:

        array: The array of tuples to filter. Each tuple is of the form (timestamp, data, metadata).

        timestamp_start: The start of the timestamp range.

        timestamp_end: The end of the timestamp range.



    Returns:

        A new array containing the tuples from the original array whose timestamps fall within the given range.

    """

    filtered_array = []

    for timestamp, data, metadata in array:

        if timestamp_start <= timestamp <= timestamp_end:

            filtered_array.append((timestamp, data, metadata))

    return filtered_array

