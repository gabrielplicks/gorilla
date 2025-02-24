import re



def replace_grace_hopper(text: str) -> str:

    """Replaces all instances of the word "grace" followed by "hopper" with "Grace Hopper".

    The function uses regular expressions to perform this replacement, allowing for variations in capitalization, punctuation, or spacing.

    Args:

        text: The input string.

    """

    pattern = re.compile(r'\bgrace\b\s*hopper\b', re.IGNORECASE)

    return pattern.sub('Grace Hopper', text)

