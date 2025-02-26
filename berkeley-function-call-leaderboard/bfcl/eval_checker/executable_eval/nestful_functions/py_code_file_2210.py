from typing import Dict, List



def find_users(graph: Dict[str, List[str]], name: str) -> Dict[str, List[str]]:

    """Finds users in the given graph who have the given name in their name.



    Args:

        graph: A dictionary representing the graph of a user's social network.

            Each key represents a user and the corresponding value is a list of

            the user's friends.

        name: A string to search for in the user's name.



    Returns:

        A dictionary where the keys are the users who have the given name in

        their name and the values are the list of friends for each user.

    """

    return {user: friends for user, friends in graph.items() if name.lower() in user.lower()}

