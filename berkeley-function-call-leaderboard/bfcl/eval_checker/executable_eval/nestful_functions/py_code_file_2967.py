def map_user_to_score(user_id: str) -> int:

    """Maps a user ID to a score based on the user's ID.



    For user IDs that start with 1, the score is the length of the ID.

    For user IDs that start with 2, the score is the length of the ID + 1.

    For all other user IDs, the score is the absolute difference between the length of the ID and 10.



    Args:

        user_id: The user ID.



    Returns:

        The score mapped to the user ID.

    """

    if user_id.startswith('1'):

        return len(user_id) + 1

    elif user_id.startswith('2'):

        return len(user_id) + 2

    else:

        return abs(len(user_id) - 10)

