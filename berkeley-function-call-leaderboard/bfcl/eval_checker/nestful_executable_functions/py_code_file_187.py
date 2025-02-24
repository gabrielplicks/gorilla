import json

from typing import Dict



def copy_values(obj1: Dict[str, str], obj2: Dict[str, str]) -> None:

    """Copies values from the first JSON object to the second one.



    Args:

        obj1: The first JSON object.

        obj2: The second JSON object.

    """

    for key in obj1:

        if key in obj1:

            obj2[key] = obj1[key]

