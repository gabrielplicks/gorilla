from typing import List, Dict



def get_names_from_log(log: List[Dict[str, str]]) -> List[str]:

    """Extracts names or usernames from a list of log entries.



    Args:

        log: A list of dictionaries representing log entries.



    Returns:

        A list of names or usernames associated with the log entries.

    """

    names = []

    for entry in log:

        if 'name' in entry:

            names.append(entry['name'])

        elif 'username' in entry:

            names.append(entry['username'])

    return names

