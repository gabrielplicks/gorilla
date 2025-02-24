import re



def parse_config(config_file: str, keys: List[str]) -> Dict[str, str]:

    """Parses a configuration file and extracts the values of a given key(s).



    Args:

        config_file: The path to the configuration file.

        keys: The keys to extract from the configuration file.



    Returns:

        A dictionary that maps each key to its value.

    """

    config = {}

    with open(config_file, 'r') as f:

        for line in f:

            match = re.search(r'^(?P<key>[a-z0-9]+)=(?P<value>.+)$', line.strip())

            if match:

                key = match.group('key')

                value = match.group('value')

                if key in keys:

                    config[key] = value



    return config

