import re



def validate_docker_compose_version(version_string: str) -> bool:

    """Validates a string as a valid Docker Compose file version.



    Args:

        version_string: The string to validate.



    Returns:

        True if the string is a valid Docker Compose file version, False otherwise.

    """

    pattern = r"^(1|2)(\.(rc|dev)\d+)?$"

    return bool(re.match(pattern, version_string))

