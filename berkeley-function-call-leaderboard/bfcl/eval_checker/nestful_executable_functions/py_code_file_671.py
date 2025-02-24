from typing import List



class User:

    def __init__(self, organization: str):

        self.organization = organization



def filter_internal_users(users: List[User]) -> List[User]:

    """Filters out internal users from a given list of users.

    Internal users belong to an organization, and the organization has a name that starts with "Internal".

    Args:

        users: A list of users.

    Returns:

        A list of internal users.

    """

    internal_users = []

    for user in users:

        if user.organization and user.organization.startswith("Internal"):

            internal_users.append(user)

    return internal_users

