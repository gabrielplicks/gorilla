import json

from typing import List, Dict



def get_all_users(data: str) -> List[str]:

    """Extracts the usernames from a JSON-formatted string containing user information.



    Args:

        data: A JSON-formatted string containing user information.



    Returns:

        A list of usernames.

    """

    users = json.loads(data)["users"]

    return [user["username"] for user in users]

