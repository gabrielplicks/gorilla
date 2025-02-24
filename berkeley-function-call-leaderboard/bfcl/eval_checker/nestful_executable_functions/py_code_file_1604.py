from typing import List, Dict, Any



def get_key_word(contacts: List[Dict[str, Any]]) -> str:

    """

    Returns the value of the `keyword` attribute within a `contact` object, which is nested within the `content` object, which is inside a list named `contacts`.

    If the `content` object is missing, the function returns an empty string.

    If the `keyword` attribute is missing or is not a string, the function returns the string `"missing"`.



    Args:

        contacts: A list of dictionaries, where each dictionary represents a contact.

    """

    contact = contacts.get("content")

    if contact and isinstance(contact, dict):

        keyword = contact.get("keyword")

        if keyword and isinstance(keyword, str):

            return keyword



    return "missing"

