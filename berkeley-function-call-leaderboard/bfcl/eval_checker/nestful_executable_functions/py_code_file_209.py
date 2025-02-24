import re



def get_issue_key(key: str) -> int:

    """Extracts the number portion of an issue key in the format of "<project>-<number>".



    Args:

        key: A string representing an issue key in the format of "<project>-<number>".



    Raises:

        ValueError: If the input format does not match the expected pattern.

    """

    match = re.search(r"^([A-Z]+)-(\d+)$", key)

    if match:

        return int(match.group(2))

    else:

        raise ValueError(f"Invalid issue key format: {key}")

