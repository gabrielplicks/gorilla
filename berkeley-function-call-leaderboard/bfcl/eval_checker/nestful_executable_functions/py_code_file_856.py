import json

from typing import Dict, Any



def get_first_name(user_data: str) -> str:

    """Returns the first name of the user from the given JSON data.



    Args:

        user_data: A JSON string containing user data.



    Returns:

        The first name of the user.

    """

    json_data: Dict[str, Any] = json.loads(user_data)

    first_name: str = json_data["first_name"]



    return first_name

