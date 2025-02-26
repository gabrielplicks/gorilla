from typing import Dict



def update_keys(input_dict: Dict[str, str]) -> Dict[str, str]:

    """Updates the keys in a dictionary based on the values.



    If a value starts with "key:", the key is updated to the substring after "key:".

    Otherwise, the original key-value pair is added to the output dictionary.



    Args:

        input_dict: The input dictionary.



    Returns:

        A new dictionary with updated key-value pairs.

    """

    output_dict = {}

    for key, value in input_dict.items():

        if isinstance(value, str) and value.startswith("key:"):

            new_key = value.split(":")[1]

            output_dict[new_key] = input_dict[key]

        else:

            output_dict[key] = value

    return output_dict

