from typing import Dict, List, Union



def build_command(args_dict: Dict[str, Union[str, List[str]]]) -> str:

    """Builds a command line string from a dictionary of command line arguments.

    Args:

        args_dict: A dictionary where the keys are the command line arguments and the values are the argument values.

    """

    command_args = []

    for key, value in args_dict.items():

        if isinstance(value, list):

            value_str = " ".join(value)

        else:

            value_str = str(value)

        command_args.append(f"--{key.replace('_', '-')} {value_str}")

    return " ".join(command_args)

