import re

from typing import Optional



def get_ec2_snapshot_id(description: str) -> Optional[str]:

    """Retrieves the snapshot ID of an Amazon Web Services (AWS) EC2 instance using the description.

    The description string contains the snapshot ID, followed by the word "Ebs".

    Args:

        description: The description string containing the snapshot ID.

    Returns:

        The snapshot ID if found, otherwise None.

    """

    match = re.search(r"snap-[a-zA-Z0-9]+", description)

    if match:

        return match.group()

    return None

