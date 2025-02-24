from typing import List, Dict



def concatenate_messages(dicts: List[Dict], status: str) -> str:

    """Concatenates all messages with a particular status and a newline for each message.



    Args:

        dicts: A list of dictionaries, each with a nested `status` key and nested `message` key.

        status: A status string.



    Returns:

        A string that concatenates all messages with that particular status and a newline for each message.

        If a message is missing, the function returns an empty string.

    """

    messages = [d['message'] for d in dicts if 'status' in d and d['status'] == status]

    return '\n'.join(messages)

