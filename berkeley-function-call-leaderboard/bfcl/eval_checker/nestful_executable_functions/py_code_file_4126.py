from typing import List, Dict



def case_insensitive_filter(dict_list: List[Dict], comparison_key: str) -> List[Dict]:

    """Filters out all the None values from a list of dictionaries.

    The function applies a case-insensitive string comparison, comparing the `key` attribute of the dictionaries with the given `comparison_key`.

    Returns the filtered list, which includes only dictionaries that match the comparison. If the comparison fails, then the dictionary is discarded.



    Args:

        dict_list: A list of dictionaries.

        comparison_key: The key to compare with the `key` attribute of the dictionaries.

    """

    filtered_list = []

    for dictionary in dict_list:

        if dictionary["key"].lower() == comparison_key.lower():

            filtered_list.append(dictionary)

    return filtered_list

