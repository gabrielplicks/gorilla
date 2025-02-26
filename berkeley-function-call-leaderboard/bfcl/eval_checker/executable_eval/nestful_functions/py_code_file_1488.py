from typing import List, Dict, Tuple



def extract_values_from_dict_list(dict_list: List[Dict[str, str]]) -> List[Tuple[str, str]]:

    """Extracts the values of 'id' and 'date' keys from a list of dictionaries and returns a list of tuples.



    Args:

        dict_list: A list of dictionaries.



    Returns:

        A list of tuples. Each tuple contains the values of 'id' and 'date' keys in the dictionary.

    """

    result = []

    for d in dict_list:

        id_value = d.get('id')

        date_value = d.get('date')

        result.append((id_value, date_value))

    return result

