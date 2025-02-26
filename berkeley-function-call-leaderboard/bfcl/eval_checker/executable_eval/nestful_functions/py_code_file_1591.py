from typing import Union



def generate_doc_id(user_id: Union[str, int], file_name: str) -> str:

    """Generates a document ID based on a given user ID and file name.

    The ID is a combination of the user ID and the file name, with a dash (-) character as the separator.

    Args:

        user_id: The user ID.

        file_name: The file name.

    """

    return f'{user_id}-{file_name}'

