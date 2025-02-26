from typing import List



def get_strings_with_prefix(words: List[str], prefix: str) -> List[str]:

    """Returns a list of strings from `words` that contain the `prefix` as a prefix, with the prefix removed.



    Args:

        words: A list of strings.

        prefix: A string to be used as a prefix.



    """

    return [word[len(prefix):] for word in words if word.startswith(prefix)]

