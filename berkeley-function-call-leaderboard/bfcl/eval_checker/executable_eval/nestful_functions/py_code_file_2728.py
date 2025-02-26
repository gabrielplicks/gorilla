from typing import List, Dict, Any



def extract_from_list(array: List[Dict[str, Any]], key: str) -> List[Any]:

    """Extracts the values associated with the specified key from a list of dictionaries.



    Args:

        array: A list of dictionaries.

        key: The key to extract values for.



    Returns:

        A list of values associated with the specified key. If a key does not exist in a dictionary, the corresponding value is `None`.

    """

    output = []

    for item in array:

        if key in item:

            output.append(item[key])

        else:

            output.append(None)

    return output

