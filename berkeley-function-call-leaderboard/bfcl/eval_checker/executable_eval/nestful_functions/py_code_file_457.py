from typing import Dict



def extract_config_file(file_path: str) -> Dict[str, str]:

    """Extracts the contents of a config file from a provided path.

    The config file is a list of key-value pairs, with each pair on a separate line.

    The key and value are separated by the symbol "=" and there may be spaces around the "=" symbol.

    If there are multiple "=" symbols on a line, the last one is used.

    The function returns a dictionary where the keys are the keys from the config file

    and the values are the corresponding values.

    Args:

        file_path: The path to the config file.

    """

    config = {}

    with open(file_path, 'r') as f:

        for line in f:

            key, value = line.split('=', maxsplit=1)

            key = key.strip()

            value = value.strip()

            config[key] = value

    return config

