import re



def string_replace(input_string: str, from_str: str, to_str: str) -> str:

    """Replaces all occurrences of a given substring in a string using regular expression matching.



    Args:

        input_string: A string containing the text where substring replacement should be performed.

        from_str: The substring to be replaced in the input string.

        to_str: The string to replace occurrences of `from_str` in the input string.

    """

    return re.sub(rf"\b{re.escape(from_str)}\b", to_str, input_string)

