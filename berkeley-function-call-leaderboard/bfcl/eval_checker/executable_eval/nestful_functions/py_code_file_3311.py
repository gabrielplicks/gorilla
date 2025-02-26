import re

from typing import List



def convert_and_append(string_to_split: str) -> List[int]:

    """Converts a comma-separated string to a list of integers and appends them to bool_list.



    Args:

        string_to_split: A comma-separated string of integers.



    Returns:

        A list of integers, where each integer is either 0 or 1 based on the corresponding

        integer in the input string.

    """

    values = re.findall('[^,]+', string_to_split)

    bool_list = [1 if int(value) % 2 else 0 for value in values]

    return bool_list

