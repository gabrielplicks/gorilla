from typing import List



def parse_range_str(range_str: str) -> List[int]:

    """Parses a string of a range of integers in the format 'start-end' and returns a list of integers within that range.



    Args:

        range_str: The input string of a range of integers in the format 'start-end'.



    Returns:

        A list of integers within the range.

    """

    start, end = range_str.split('-')

    start = int(start)

    end = int(end)

    range_list = []

    for i in range(start, end + 1):

        range_list.append(i)

    return range_list

