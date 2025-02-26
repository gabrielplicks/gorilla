from typing import Tuple



class User:

    def __init__(self, name: str, email: str, role: str, is_active: bool):

        self.name = name

        self.email = email

        self.role = role

        self.is_active = is_active



def format_user_info(user: User) -> str:

    """Formats user information into a string.



    Args:

        user: The User instance to format.



    Returns:

        A formatted string containing the user's name, email, role, and active status.

    """

    return f"{user.name} <{user.email}> ({user.role}): is_active={user.is_active}"

