from typing import List, Dict



def group_users_by_attribute(users: List[Dict], attribute: str) -> (List[Dict], List[Dict]):

    """Groups users into two separate groups based on a specific attribute.



    Args:

        users: A list of dictionaries representing users.

        attribute: The attribute to check for in each user dictionary.



    Returns:

        A tuple containing two lists:

        - The first list contains the users with the attribute.

        - The second list contains the users without the attribute.

    """

    users_with_attribute = []

    users_without_attribute = []



    for user in users:

        if attribute in user:

            users_with_attribute.append(user)

        else:

            users_without_attribute.append(user)



    return users_with_attribute, users_without_attribute

