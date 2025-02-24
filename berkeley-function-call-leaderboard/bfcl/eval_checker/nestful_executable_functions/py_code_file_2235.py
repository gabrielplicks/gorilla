from typing import List



def command_line(args: List[str]) -> str:

    """Creates a string representation of the full command-line invocation of a script.



    Args:

        args: A list of arguments to be included in the command-line invocation.



    Returns:

        A string containing the full command-line invocation of the script.

    """

    command = ["python3", "myscript.py"]

    for arg in args:

        command.append(arg)

    return " ".join(command)

