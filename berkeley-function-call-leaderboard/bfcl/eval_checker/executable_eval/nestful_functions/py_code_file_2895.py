from typing import Dict, List, Any



def convert_list_of_dictionaries(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:

    """Converts a list of dictionaries into a dictionary.

    The keys of the new dictionary are the "string" keys from the list of dictionaries,

    and the values are the "value" keys from the list of dictionaries.

    Args:

        list_of_dicts: A list of dictionaries. Each dictionary has two keys: a string key and a value.

    """

    return {d["string"]: d["value"] for d in list_of_dicts}

