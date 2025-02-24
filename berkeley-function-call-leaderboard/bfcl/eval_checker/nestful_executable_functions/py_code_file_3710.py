from typing import List, Dict



def extract_prefix(input_list: List[Dict[str, int]]) -> List[Dict[str, int]]:

    """Creates a new list of dictionaries containing only the prefix of each dictionary.

    Args:

        input_list: A list of dictionaries representing a complex data structure.

    """

    output_list = []

    for dictionary in input_list:

        prefix = {key: value for key, value in dictionary.items() if key == 'a'}

        output_list.append(prefix)

    return output_list

