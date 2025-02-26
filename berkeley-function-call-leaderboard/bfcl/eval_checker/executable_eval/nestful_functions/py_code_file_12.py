from typing import Dict, List, Union



def parse_and_assign(data: List[str]) -> Dict[str, Union[str, List[str], None]]:

    """Parses a list of strings and creates a dictionary where each key is a key name and each value is the corresponding value.



    Args:

        data: A list of strings of the form `key1:value1,key2:value2,key3:value3,…`.



    Returns:

        A dictionary where each key is a key name and each value is the corresponding value.

    """

    result = {}

    for string in data:

        pairs = string.split(",")

        for pair in pairs:

            key, value = pair.split(":")

            if not value:

                value = None

            elif "|" in value:

                value = value.split("|")

            result[key] = value

    return result

