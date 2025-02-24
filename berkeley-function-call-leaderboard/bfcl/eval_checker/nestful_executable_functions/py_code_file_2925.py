import configparser

from typing import Dict, Any



def read_config(config_file: str) -> Dict[str, Dict[str, Any]]:

    """Reads a configuration file and returns a dictionary containing the user information and database information.



    Args:

        config_file: The path to the configuration file.



    Returns:

        A dictionary containing the user information and database information.

    """

    config = configparser.ConfigParser()

    config.read(config_file)



    config_dict: Dict[str, Dict[str, Any]] = {}



    for section in config.sections():

        config_dict[section] = {}

        for option in config.options(section):

            config_dict[section][option] = config.get(section, option)



    return config_dict

