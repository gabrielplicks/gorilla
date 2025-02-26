from typing import List, Dict, Any



def generate_descriptors(input_list: List[Dict[str, Any]], attributes: List[str]) -> List[str]:

    """Generates a list of strings based on the values of a specific set of attributes.

    Args:

        input_list: A list of dictionaries.

        attributes: A list of attribute names.

    Returns:

        A list of strings, where each string is a concatenation of an attribute name and its corresponding value in the dictionary.

    """

    return [f"{attribute}={item[attribute]}" for item in input_list for attribute in attributes]

