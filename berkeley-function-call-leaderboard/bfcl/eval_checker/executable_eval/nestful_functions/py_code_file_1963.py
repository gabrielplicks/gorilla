from datetime import datetime

from typing import Any

import os



def create_dynamic_file_path(instance: Any, timestamp: datetime) -> str:

    """Creates a dynamic file path based on a given timestamp and instance object attributes.

    The file path includes the instance name, timestamp, and the file extension .txt.



    Args:

        instance: The object containing the instance name.

        timestamp: The timestamp to be included in the file path.

    """

    instance_name = instance.name

    file_name = f"{instance_name}_{timestamp.strftime('%Y-%m-%d %H:%M:%S')}.txt"

    file_path = os.path.join(instance_name, file_name)

    return file_path

