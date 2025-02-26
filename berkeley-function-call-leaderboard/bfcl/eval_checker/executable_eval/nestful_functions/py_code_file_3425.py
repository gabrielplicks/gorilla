import re



def parse_and_replace(text: str, mapping: dict) -> str:

    """

    Performs a step-by-step replacement of keys in `mapping` in the order of their lengths.

    Args:

        text: The input string to perform the replacements on.

        mapping: A dictionary containing the keys to replace and their corresponding values.

    """

    sorted_keys = sorted(mapping, key=lambda x: len(x), reverse=True)

    pattern = '|'.join(sorted_keys)

    return re.sub(pattern, lambda x: mapping[x.group()], text)

