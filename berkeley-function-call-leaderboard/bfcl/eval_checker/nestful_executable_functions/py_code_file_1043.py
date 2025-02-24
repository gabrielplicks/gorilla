from typing import Union



def check_command_type(command: Union[int, str]) -> bool:

    """Determines whether a given command is a request or a response.

    Args:

        command: The command to check.

    """

    return (command & 0b10000001) == 0b10000001

