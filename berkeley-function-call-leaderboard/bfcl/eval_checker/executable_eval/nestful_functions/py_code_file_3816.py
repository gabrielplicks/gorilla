import sys

from typing import List



def check_command_args() -> bool:

    """

    Checks whether the command line argument starts with `python`.

    If it does, prints a message and returns `True`; otherwise, prints a different message and returns `False`.

    If no command line arguments are provided, raises an exception.



    Args:

        None



    Raises:

        Exception: If no command line arguments are provided.

    """

    if len(sys.argv) == 1:

        raise Exception("No command line arguments provided")

    if sys.argv[1].startswith("python"):

        print("Command line argument starts with 'python'")

        return True

    print("Command line argument does not start with 'python'")

    return False

