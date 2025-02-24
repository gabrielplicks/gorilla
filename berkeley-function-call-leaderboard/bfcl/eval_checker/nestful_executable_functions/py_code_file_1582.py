from typing import Any



def logout_session(user: Any) -> bool:

    """Logs out the current session and returns true if successful, false if unsuccessful.



    Args:

        user: The user object containing the current session.



    Returns:

        True if the logout was successful, False otherwise.

    """

    result = user.session.logout()

    if result:

        return True

    else:

        return False

