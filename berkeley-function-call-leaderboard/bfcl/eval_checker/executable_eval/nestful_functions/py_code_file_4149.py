from typing import Dict



def encode_special_characters(string: str) -> str:

    """Encodes special characters in a string to HTML entities.



    Args:

        string: The input string.



    Returns:

        The string with special characters encoded to HTML entities.

    """

    special_characters: Dict[str, str] = {

        '<': '&lt;',

        '>': '&gt;',

        '&': '&amp;',

        '"': '&quot;',

        "'": '&apos;'

    }



    return "".join(

        [

            special_characters.get(c, c)

            for c in string

        ]

    )

