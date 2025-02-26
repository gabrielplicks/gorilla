import re



def validate_twitter_username(username: str) -> bool:

    """Validates a Twitter username string.



    Args:

        username: The Twitter username to validate.



    Returns:

        True if the username is valid, False otherwise.

    """

    regex = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,14}\S$")

    if regex.match(username):

        return True

    else:

        return False

