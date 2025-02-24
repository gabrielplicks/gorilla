from typing import Dict



def filter_dict_by_ids(original_dict: Dict[str, int], ids_to_filter: List[str]) -> Dict[str, int]:

    """Filters a dictionary based on a list of item IDs.



    Args:

        original_dict: The dictionary to filter.

        ids_to_filter: The list of IDs to filter.



    Returns:

        A dictionary containing the filtered items.

    """

    filtered_dict = {}



    for id in ids_to_filter:

        value = original_dict.get(id)

        if value is not None:

            filtered_dict[id] = value



    return filtered_dict

