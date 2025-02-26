from typing import Dict



def generate_configuration_file(filename: str, properties: Dict[str, str]) -> None:

    """Generates a configuration file from a given filename and a dictionary of properties.



    The function writes the properties to the file in a human-readable format,

    with one property per line, and the property name and value separated by an equal sign.



    Args:

        filename: The name of the configuration file.

        properties: A dictionary of key-value pairs representing the properties.

    """

    with open(filename, "w") as file:

        for key, value in properties.items():

            file.write(f"{key}={value}\n")

