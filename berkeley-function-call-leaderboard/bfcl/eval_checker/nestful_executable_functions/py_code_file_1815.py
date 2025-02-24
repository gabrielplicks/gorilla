import json

from typing import Any, Dict



def get_max_height_helper(obj: Dict[str, Any], height: int) -> int:

    """Calculates the maximum height of a JSON object.



    Args:

        obj: The JSON object.

        height: The current height.

    """

    if not 'children' in obj:

        return height

    return max(get_max_height_helper(child, height + 1) for child in obj['children'])



def get_max_height(data: str) -> int:

    """Calculates the maximum height of a JSON object.



    Args:

        data: The JSON object as a string.

    """

    root = json.loads(data)

    return get_max_height_helper(root, 1)

