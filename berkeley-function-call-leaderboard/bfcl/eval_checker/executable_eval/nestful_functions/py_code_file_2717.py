from typing import Dict



class User:

    def __init__(self, id: int, username: str, email: str, is_active: bool):

        self.id = id

        self.username = username

        self.email = email

        self.is_active = is_active



def get_user_data(user: User) -> Dict[str, str]:

    """

    Returns a dictionary containing the user's id, username, email, and a string version of the user's is_active attribute.

    Args:

        user: The User object.

    """

    user_dict = vars(user)  # Get a dictionary representation of the User object

    return {

        "id": str(user_dict["id"]),

        "username": user_dict["username"],

        "email": user_dict["email"],

        "is_active": "active" if user_dict["is_active"] else "inactive",

    }

