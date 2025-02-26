import re



def replace_between_brackets(s: str, t: str) -> str:

    """Replaces the substring between the first pair of square brackets in a string with another string.



    Args:

        s: The input string.

        t: The replacement string.



    Returns:

        The updated string.

    """

    matches = re.findall(r"\[([^\[\]]+)\]", s)

    if matches:

        updated_s = re.sub(rf"\[({matches[0]})\]", rf"[{t}]", s, count=1)

        return updated_s

    else:

        return s

