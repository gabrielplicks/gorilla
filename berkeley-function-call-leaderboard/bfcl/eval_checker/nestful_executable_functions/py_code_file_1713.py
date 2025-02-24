from typing import List, Dict



def get_keys_as_string_array(obj: Dict) -> List[str]:

    """Converts the keys of an object to a string array and returns it.



    Args:

        obj: The object with a `keys()` method.



    Returns:

        A list of strings representing the keys of the object.

    """

    return list(map(str, obj.keys()))

