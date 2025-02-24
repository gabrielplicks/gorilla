from typing import List



def create_database_key(input_string: str) -> str:

    """Creates a database key from a given string.

    The key is in the form of 'key_a-b-c', where each 'a', 'b', 'c', etc. is a character in the input string.

    Args:

        input_string: The input string.

    """

    characters: List[str] = list(input_string)

    characters.sort()

    characters_joined: str = '-'.join(characters)

    database_key: str = f'key_{characters_joined}'

    return database_key

