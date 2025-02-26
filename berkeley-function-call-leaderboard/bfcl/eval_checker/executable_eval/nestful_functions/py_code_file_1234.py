from typing import Dict, Any



def replace_variables(message: str, variables: Dict[str, Any]) -> str:

    """

    Replaces variables in a message with their corresponding values.



    Args:

        message: The message containing variables to be replaced.

        variables: A dictionary containing variable names as keys and their corresponding values.



    Returns:

        The message with variables replaced by their corresponding values.

    """

    try:

        if not isinstance(message, str):

            raise TypeError("Message must be a string")

        if not isinstance(variables, dict):

            raise TypeError("Variables must be a dictionary")

        for key, value in variables.items():

            if not isinstance(key, str):

                raise TypeError("Keys must be strings")

        return message.format(**variables)

    except Exception as e:

        print(f"Error: {e}")

        return "Error occurred while replacing variables"

