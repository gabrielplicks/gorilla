from typing import Optional



def replace_prefix(string: str, old_prefix: Optional[str], new_prefix: Optional[str]) -> str:

    """Replaces the old prefix in a string with a new prefix.



    Args:

        string: The string to replace the prefix in.

        old_prefix: The prefix to replace.

        new_prefix: The new prefix to replace the old prefix with.



    Returns:

        The string with the old prefix replaced with the new prefix.

    """

    if not old_prefix:

        return string

    if string.startswith(old_prefix):

        return new_prefix + string[len(old_prefix):]

    else:

        return string

