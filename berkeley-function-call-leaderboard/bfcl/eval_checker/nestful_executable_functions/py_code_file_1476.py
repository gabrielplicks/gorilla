from typing import List, Dict



def get_active_users(users: List[Dict[str, str]]) -> List[str]:

    """Returns a list of users' names whose `is_active` attribute is `True`.



    Args:

        users: A list of dictionaries, each representing a user.



    Returns:

        A list of users' names.

    """

    return [user['name'] for user in users if user.get('is_active')]

