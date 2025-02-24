import re



def parse_user_ids(user_ids_str: str) -> list[int]:

    """

    Parses a comma-separated string of user IDs into a list of integers.

    For example, given user_ids_str = '001,002,003', this function

    returns [1, 2, 3].

    Args:

        user_ids_str: A string containing a list of comma-separated user IDs.

    """

    return [int(user_id) for user_id in re.findall(r'\d+', user_ids_str)]

