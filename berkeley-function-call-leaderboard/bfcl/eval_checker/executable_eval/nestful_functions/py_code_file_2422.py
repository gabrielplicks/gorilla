from typing import List, Union



def get_range_description(table: List[List[Union[str, None]]], range_string: str) -> Union[str, None]:

    """

    Returns the corresponding description for a given range string, or None if the range string is not present in the table.



    Args:

        table: A list of lists, where each inner list corresponds to a row in the table. Each row contains two elements: the range string and the description string.

        range_string: A string in the form of "x-y" representing a range.

    """

    range_strings = [row[0] for row in table]

    if range_string in range_strings:

        index = range_strings.index(range_string)

        return table[index][1]

    return None

