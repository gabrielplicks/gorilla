from typing import List, Dict



def parse_key_val_string(input_string: str) -> List[Dict[str, str]]:

    """Parses a `key=val` format string into a list of dictionaries.

    Each dictionary represents a single key-value pair in the format string.

    The list contains all the dictionaries.



    Args:

        input_string: The input string in `key=val` format.



    Returns:

        A list of dictionaries, where each dictionary represents a single key-value pair.

    """

    key_val_pairs = input_string.split(',')

    dict_list = []

    for pair in key_val_pairs:

        key, val = pair.split('=')

        dict_list.append({'key': key, 'val': val})

    return dict_list

