import json

from typing import Any



def get_unique_ip_count(file_name: str) -> int:

    """

    Returns the number of unique values in the `ip` key within a JSON file.



    Args:

        file_name: The name of the JSON file.



    Returns:

        The number of unique values in the `ip` key.

    """

    with open(file_name, 'r') as f:

        data: list[Any] = json.loads(f.read())

        ip_values = [item['ip'] for item in data if 'ip' in item]

        unique_values = set(ip_values)

        return len(unique_values)

