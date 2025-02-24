import re



def is_semantic_version(version_str: str) -> bool:

    """Checks if a given string matches the pattern of a Semantic Version.



    Args:

        version_str: The string to check.



    Returns:

        True if the string matches the pattern, False otherwise.

    """

    pattern = r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$"

    match = re.match(pattern, version_str)

    if not match:

        return False

    major, minor, patch = match.groups()

    return major.isdigit() and minor.isdigit() and patch.isdigit()

