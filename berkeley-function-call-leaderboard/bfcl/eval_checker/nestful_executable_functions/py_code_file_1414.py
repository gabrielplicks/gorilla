import re

from typing import Dict, Union



usernames: Dict[str, str] = {

    "Alice": "https://api.github.com/users/Alice",

    "Bob": "https://api.github.com/users/Bob",

    "Charlie": "https://api.github.com/users/Charlie"

}



def get_github_api_url(username: str) -> Union[str, None]:

    """

    Returns the corresponding GitHub API URL for a given GitHub username, or None if the username is not found.

    Args:

        username: The GitHub username.

    """

    if not re.match(r"^[a-zA-Z0-9-]+$", username):

        raise ValueError("Username must consist of alphanumeric characters and dashes.")



    return usernames.get(username, None)

