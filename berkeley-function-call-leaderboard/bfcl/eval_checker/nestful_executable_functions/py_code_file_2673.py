from typing import List, Dict



def merge_duplicate_entries(input_list: List[Dict], keys: str) -> List[Dict]:

    """Merges duplicate entries in a list of dictionaries.



    Args:

        input_list: A list of dictionaries.

        keys: The key to merge on.



    Returns:

        A list of dictionaries with merged entries.

    """

    merged_dict = {}

    for entry in input_list:

        key = entry[keys]

        value = entry['value']

        if key in merged_dict:

            merged_dict[key].append(value)

        else:

            merged_dict[key] = [value]

    final_list = []

    for key, value in merged_dict.items():

        final_list.append({keys: key, 'value': value})

    return final_list

