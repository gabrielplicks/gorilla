import re



def replace_value(string: str, key_value: str, new_value: str) -> str:

    """Replaces all instances of a specified key value in a string with a new value using regular expressions.



    Args:

        string: The original string.

        key_value: The key value to replace.

        new_value: The new value to replace with.

    """

    pattern = re.compile(r"\b" + re.escape(key_value) + r"\b")

    result = pattern.sub(new_value, string)

    return result

