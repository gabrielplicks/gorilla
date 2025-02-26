import re

from typing import Dict



def write_filtered_data(data: Dict[str, any], filter_pattern: str) -> Dict[str, any]:

    """Writes filtered data from a dictionary based on a filter pattern.



    Args:

        data: The dictionary of data to filter.

        filter_pattern: The filter pattern to match against the keys of the data.

    """

    filtered_data = {}

    for key, value in data.items():

        if re.match(filter_pattern, key):

            filtered_data[key] = value



    return filtered_data

