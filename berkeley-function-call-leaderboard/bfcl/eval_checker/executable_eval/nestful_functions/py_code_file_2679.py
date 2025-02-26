from typing import Dict, List



def get_keys_with_hello(d: Dict[str, str]) -> List[str]:

    """Returns a list of keys that contain the string "hello" as a substring.



    Args:

        d: The dictionary to search for keys containing "hello".



    Returns:

        A list of keys that contain the string "hello" as a substring.

    """

    keys_with_hello = []

    for key in d:

        if "hello" in key:

            keys_with_hello.append(key)

    return keys_with_hello

