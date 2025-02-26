from typing import Dict



def convert_dict_keys(d: Dict[str, any], key_map: Dict[str, str]) -> Dict[str, any]:

    """Converts the keys of a dictionary according to a key map.



    Args:

        d: The dictionary to convert.

        key_map: A dictionary mapping old keys to new keys.

    """

    new_dict = {}

    for key, value in d.items():

        new_dict[key_map[key]] = value

    return new_dict

