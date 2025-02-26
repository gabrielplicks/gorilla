import json

from typing import List



def serialize_list_to_json(args: List[str]) -> str:

    """Serializes a list of paired arguments into a JSON string, where the odd-indexed arguments are keys and the even-indexed arguments are values.



    Args:

        args: The list of paired arguments.



    Returns:

        The JSON string.

    """

    return json.dumps({k: v for k, v in zip(args[::2], args[1::2])})

