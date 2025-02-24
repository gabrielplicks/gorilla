import re

from typing import Dict



def load_config_file(config_file: str) -> Dict[str, str]:

    """Loads a configuration file and returns a dictionary representing the configurations.



    Each line in the configuration file is in the format of `<key>:<value>`.

    The function ignores lines that do not follow this format.



    Args:

        config_file: The path to the configuration file.



    Returns:

        A dictionary representing the configurations.

    """

    config_dict = {}



    with open(config_file, 'r') as file:

        for line in file:

            line = line.strip()  # Remove leading and trailing whitespace

            match = re.search(r'^(\w+): (.+)$', line)

            if match:

                key, value = match.groups()

                config_dict[key] = value



    return config_dict

