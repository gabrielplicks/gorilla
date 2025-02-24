from typing import Dict, Optional



def create_user_info(name: Optional[str] = None, age: Optional[int] = None, favorite_color: Optional[str] = None) -> Dict[str, str]:

    """Creates a dictionary with user information.



    Args:

        name: The name of the user. Defaults to "Anonymous".

        age: The age of the user. Defaults to -1.

        favorite_color: The favorite color of the user. Defaults to "Unknown".

    """

    user_info = {}



    if name is None:

        name = "Anonymous"

    user_info["name"] = name



    if age is None:

        age = -1

    user_info["age"] = age



    if favorite_color is None:

        favorite_color = "Unknown"

    user_info["favorite_color"] = favorite_color



    return user_info

