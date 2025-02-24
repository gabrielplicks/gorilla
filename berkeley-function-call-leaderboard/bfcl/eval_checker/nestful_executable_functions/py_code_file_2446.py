from collections import defaultdict

from typing import List, Dict



def format_interactions(data: List[List[str]]) -> Dict[str, List[str]]:

    """Converts a list of user-item interactions into a dictionary whose keys are the users and the values are the corresponding items.



    Args:

        data: A list of user-item interactions. Each interaction is represented as a list of two strings: the user ID and the item ID.



    Returns:

        A dictionary that maps user IDs to a list of unique item IDs.

    """

    user_items = defaultdict(list)

    for user_id, item_id in data:

        if item_id not in user_items[user_id]:

            user_items[user_id].append(item_id)

    return user_items

