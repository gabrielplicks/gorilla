from typing import List, Tuple



def get_min_distance(users: List[Tuple[str, float]]) -> str:

    """Returns the name of the user with the minimum distance.

    Args:

        users: A list of tuples. Each tuple has two elements: the user's name and the user's distance from a certain location.

    """

    min_user = min(users, key=lambda user: user[1])

    return min_user[0]

