from typing import List, Dict, Tuple



def transform_dict_list(input_list: List[Dict[str, str]], keys: List[str]) -> List[Tuple[str, str]]:

    """Transforms a list of dictionaries into a list of keys and a list of values.



    Args:

        input_list: A list of dictionaries.

        keys: A list of keys.



    Returns:

        A list of tuples containing the values corresponding to the keys in the input dictionaries.

    """

    new_list = []

    for d in input_list:

        values = tuple(d.get(key) for key in keys)

        new_list.append(values)

    return new_list

