from typing import List, Dict



def get_friend_names(d: Dict) -> List[str]:

    """Returns a list of names of all the user's friends from a given JSON-like dictionary object `d`.



    Args:

        d: A JSON-like dictionary object.



    Returns:

        A list of names of all the user's friends.

    """

    if 'friends' in d and isinstance(d['friends'], list):

        return [friend['name'] for friend in d['friends'] if 'name' in friend]

    else:

        return []

