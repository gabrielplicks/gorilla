from typing import Dict, List



def find_keys_with_hello(dictionary: Dict[str, str]) -> List[str]:

    """Finds keys in a dictionary that contain the word "Hello" in their values.



    Args:

        dictionary: A dictionary where the values are strings.



    Returns:

        A list of keys that contain the word "Hello" in their values.

    """

    keys_with_hello = []

    for key, value in dictionary.items():

        if "Hello" in value:

            keys_with_hello.append(key)

    return keys_with_hello

