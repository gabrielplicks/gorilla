from collections import defaultdict

from operator import itemgetter

from typing import List, Dict



def combine_values(input_list: List[Dict[str, str]]) -> List[Dict[str, List[str]]]:

    """Combines the values of dictionaries with the same key into a list.



    Args:

        input_list: A list of dictionaries, each dictionary has a key and a list of values.



    Returns:

        A list of dictionaries, each dictionary has a key and a list of combined values.

    """

    sorted_list = sorted(input_list, key=itemgetter('key'))

    grouped = defaultdict(list)

    for item in sorted_list:

        grouped[item['key']].extend(item['values'])

    output_list = []

    for key, values in grouped.items():

        output_list.append({'key': key, 'values': values})



    return output_list

