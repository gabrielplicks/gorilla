from typing import List



def build_key(key_parts: List[str], prefix: str = '') -> str:

    """Concatenates the elements from key_parts with the prefix at the beginning.



    Args:

        key_parts: A non-empty list of strings.

        prefix: An optional string to be added at the beginning.



    Returns:

        A string that is the concatenation of the elements from key_parts, with the prefix at the beginning.

    """

    assert isinstance(key_parts, list) and len(key_parts) > 0 and all(isinstance(x, str) for x in key_parts), "key_parts must be a non-empty list of strings"

    assert isinstance(prefix, str), "prefix must be a string"

    key_concat = ''.join([prefix] + key_parts)

    return key_concat

