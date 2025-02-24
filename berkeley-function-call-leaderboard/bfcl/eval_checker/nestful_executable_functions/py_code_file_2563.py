from typing import List, Tuple, Dict, Any



def pairs_to_dicts(pairs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:

    """Converts a list of pairs into a list of dictionaries.



    Args:

        pairs: A list of pairs. Each pair is a tuple of a string and a value.



    Returns:

        A list of dictionaries. Each dictionary is a key-value pair of the

        corresponding pair in the input list.

    """

    output = []

    for key, value in pairs:

        if any(key in d for d in output):

            d[key] = value

        else:

            output.append({key: value})

    return output

