from typing import List, Dict



def get_values_from_dict_list(dict_list: List[Dict], string: str) -> List:

    """Returns a list of the values associated with the given string in the dictionaries.

    Args:

        dict_list: A list of dictionaries.

        string: A string to search for in the dictionaries.

    """

    values = []

    for d in dict_list:

        if string in d:

            values.append(d[string])

    return values

