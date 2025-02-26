from typing import List, Dict



class User:

    id: int

    name: str

    username: str



def get_valid_usernames(users: List[User]) -> Dict[str, User]:

    """Returns a dictionary of valid usernames and their corresponding user objects.



    Valid usernames are those with a non-zero ID and a non-empty name field.



    Args:

        users: A list of user objects.



    Returns:

        A dictionary of valid usernames and their corresponding user objects.

    """

    return {

        user.username: user

        for user in [

            user for user in users

            if user.id != 0 and user.name

        ]

    }

