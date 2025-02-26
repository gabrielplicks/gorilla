from typing import Dict, List, Union



def translate_to_dict(data: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, List[str]]:

    """Translates a list of dictionaries into a new dictionary.



    Each key in the new dictionary is the value of the "name" field in the input data,

    and each value is a list of the values of the "value" field.

    If the "name" field is empty or missing, the corresponding dictionary is discarded.



    Args:

        data: A list of dictionaries.



    Returns:

        A new dictionary.

    """

    new_dict = {}

    for d in data:

        name = d.get('name')

        if name:

            new_dict[name] = d.get('value', [])

    return new_dict

