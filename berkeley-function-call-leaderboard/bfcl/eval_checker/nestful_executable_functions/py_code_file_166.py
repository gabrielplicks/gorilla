from typing import Dict, List, Union



def list_of_strings(config: Dict[str, Union[str, List[str]]]) -> List[str]:

    """

    Returns a list of strings based on the configuration provided.

    Args:

        config: A dictionary containing the configuration.

    Raises:

        ValueError: If the configuration is missing the 'type' key or has an invalid value.

    """

    if config.get("type") == "type_1":

        return [tag for tag in config.get("tags", [])]

    elif config.get("type") == "type_2":

        return [f"{tag}\n" for tag in config.get("tags", [])]

    elif config.get("type") == "type_3":

        delimiter = config.get("delimiter", "")

        return [tag + delimiter for tag in config.get("tags", [])]

    else:

        raise ValueError("Invalid config: missing 'type' or invalid value.")

