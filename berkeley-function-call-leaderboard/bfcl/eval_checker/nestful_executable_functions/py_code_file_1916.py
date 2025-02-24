from typing import Dict



EMOJI_MAPPING: Dict[str, str] = {'a': '😀', 'b': '😁', 'c': '😂'}



def convert_to_emojis(s: str) -> str:

    """Converts a string to emojis using a dictionary lookup.



    Args:

        s: The input string to convert to emojis.



    Returns:

        The input string represented as emojis.

    """

    emojis = []

    for c in s:

        emoji = EMOJI_MAPPING.get(c, f'U+{ord(c):04X}')

        emojis.append(emoji)

    return ''.join(emojis)

