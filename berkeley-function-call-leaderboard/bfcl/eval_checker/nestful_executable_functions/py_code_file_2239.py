from typing import Dict



def update_env_value(env: Dict[str, str], key: str, value: str) -> None:

    """

    Updates the value of a specific key in a dictionary representing an environment object.

    If the key is not found in the dictionary, add it as a new entry.



    Args:

        env: The dictionary representing the environment object.

        key: The key to update or add.

        value: The new value to set for the key.

    """

    for k, v in env.items():

        if k == key:

            v = value

            break

    else:

        env[key] = value

