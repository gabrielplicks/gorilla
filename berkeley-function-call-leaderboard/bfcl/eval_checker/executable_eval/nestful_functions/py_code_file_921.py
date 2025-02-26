import ast

from typing import Dict, List, Union



def convert_values(config_file: str, names: List[str], data_types: List[type]) -> Dict[str, Union[str, int, float]]:

    """Reads a config file and returns a dictionary with the converted values.



    Args:

        config_file: The path to the config file.

        names: A list of names to extract from the config file.

        data_types: A list of data types corresponding to the names.



    Returns:

        A dictionary with the converted values.

    """

    result = {}

    with open(config_file, 'r') as f:

        config = f.read()

    for name, data_type in zip(names, data_types):

        value = config.split(f"{name} = ")[1].split("\n")[0]

        try:

            result[name] = ast.literal_eval(value)

        except ValueError:

            result[name] = value

    return result

