from typing import List, Dict



def make_user_dict(ids: List[int], names: List[str], emails: List[str]) -> Dict[int, dict]:

    """

    Make a dictionary of user objects from three lists of ids, names, and emails.



    Args:

        ids: A list of user ids.

        names: A list of user names.

        emails: A list of user emails.



    Returns:

        A dictionary of user objects, where the keys are the user ids and the values are dictionaries containing the user id, name, and email.

    """

    return {

        id: {"id": id, "name": name, "email": email}

        for id, name, email in zip(ids, names, emails)

    }

