from typing import List, Dict



def search_user_by_name(name: str, user_list: List[Dict[str, str]]) -> List[Dict[str, str]]:

    """Searches for users by name in a user list.



    Args:

        name: The name of the user to be searched.

        user_list: A list of users. Each user is represented as a dictionary with keys 'name', 'id', and 'age'.



    Returns:

        A list of users that match the given name. If no user is found, returns an empty list.

    """

    matching_users = filter(lambda user: user['name'] == name, user_list)

    return list(matching_users)

