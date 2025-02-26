import unicodedata



def get_unicode_character_info(char: str) -> str:

    """

    Returns a string containing the name of the character, along with its encoding in UTF-8.



    Args:

        char: The Unicode character.

    """

    code_point = ord(char)

    name = unicodedata.name(char)

    normalized_char = unicodedata.normalize('NFC', char)

    utf8_encoding = normalized_char.encode('utf-8')

    return f'{name} (U+{code_point:04X}, UTF-8: {" ".join(["%02X" % b for b in utf8_encoding])})'

