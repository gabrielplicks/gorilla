import re



def is_propositional_variable(s: str) -> bool:

    """Checks if a string is a propositional variable.



    A propositional variable is a string that matches the regular expression

    `[A-Z][a-z_0-9]*`, where the first character is a capital letter and the

    remaining characters are lowercase letters, underscores, or digits.



    Args:

        s: The string to check.



    Returns:

        True if the string is a propositional variable, False otherwise.

    """

    pattern = r'^[A-Z][a-z_0-9]*$'

    return re.match(pattern, s) is not None

