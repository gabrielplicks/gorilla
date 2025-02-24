from typing import List



def extract_program_name(args: List[str]) -> str:

    """Extracts the name of the program from the command line arguments.



    Args:

        args: A list of command line arguments.



    Returns:

        The name of the program as a string.

    """

    return args[0]

