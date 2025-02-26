import re

from typing import Dict, Pattern, Match



def replace_escaped_characters(input_str: str) -> str:

    """

    Replaces escaped characters in a string with their corresponding symbols.



    Args:

        input_str: The input string.



    Returns:

        A new string with escaped characters replaced by their corresponding symbols.

    """

    escape_map: Dict[str, str] = {

        '\\n': '\n',

        '\\t': '\t',

        '\\\\': '\\',

        '\\\'': '\'',

        '\\"': '"'

    }

    pattern: Pattern[str] = re.compile('|'.join(escape_map.keys()))

    def replace_escape(match: Match[str]) -> str:

        return escape_map[match.group()]

    return pattern.sub(replace_escape, input_str) if pattern.search(input_str) else input_str

