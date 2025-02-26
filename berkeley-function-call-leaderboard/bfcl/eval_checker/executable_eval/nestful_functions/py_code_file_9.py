from typing import Union



def check_user_privilege(user_id: Union[int, None]) -> Union[str, None]:

    """Checks the user's privilege level based on their ID.



    Args:

        user_id: The ID of the user.



    Returns:

        The privilege level of the user as a string, or None if the user ID is invalid or not provided.

    """

    if user_id is None or user_id <= 0:

        return None



    if user_id < 100:

        return "superuser"

    elif user_id < 10000:

        return "administrator"

    else:

        return "user"

