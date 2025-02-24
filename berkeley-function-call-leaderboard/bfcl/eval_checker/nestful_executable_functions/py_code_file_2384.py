from typing import Dict, Any



def prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:

    """

    Adds a prefix to each key in the input dictionary and returns a new dictionary.

    Args:

        d: The input dictionary.

        prefix: The prefix to be added to each key in the input dictionary.

    """

    return dict(map(lambda item: (f"{prefix}{item[0]}", item[1]), d.items()))

