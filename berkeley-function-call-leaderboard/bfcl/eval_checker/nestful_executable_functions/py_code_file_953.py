import re

from typing import Dict



def convert_dict_keys_to_camel_case(input_dict: Dict[str, Any]) -> Dict[str, Any]:

    """

    Converts the keys of a dictionary to camel case.



    Args:

        input_dict: The dictionary to be transformed.



    Returns:

        A new dictionary with the transformed keys and the same values as the input dictionary.

    """

    return {re.sub(r'_(\w)', lambda m: m.group(1).upper(), key) if '_' in key else key: value for key, value in input_dict.items()}

