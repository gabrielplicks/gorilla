from typing import List



def filter_command_args(command_args: List[str], available_args: List[str]) -> List[str]:

    """Filters command arguments by checking if they are present in the available arguments.

    Args:

        command_args: The list of command arguments.

        available_args: The list of available arguments.

    """

    return [arg for arg in command_args if arg in available_args]

