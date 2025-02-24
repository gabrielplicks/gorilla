from typing import Tuple



def parse_container_image_name(image_name: str) -> Tuple[str, str, str]:

    """Parses a container image name and returns its parts.



    Args:

        image_name: A string of the format <namespace>/<repository>:<tag>.



    Returns:

        A tuple of 3 elements:

        - namespace: a string, e.g., docker.io

        - repository: a string, e.g., library

        - tag: a string, e.g., nginx



    Raises:

        ValueError: If the input is invalid.

    """

    namespace, repository, tag = image_name.split('/')

    if not namespace or not repository:

        raise ValueError("Invalid image name: missing namespace or repository")

    if len(tag.split(':')) > 2:

        raise ValueError("Invalid image name: invalid tag format")

    return namespace, repository, tag

