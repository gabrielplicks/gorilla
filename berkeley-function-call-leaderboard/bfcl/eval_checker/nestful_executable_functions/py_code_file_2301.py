import argparse

from typing import Dict



def parse_command_line_options() -> Dict[str, str]:

    """Parses command-line options for a simple command-line tool.



    Args:

        None



    Returns:

        A dictionary with the parsed arguments, including the default values if applicable.



    Raises:

        ValueError: If one or more arguments are missing.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default='in.txt')

    parser.add_argument('--output', default='out.txt')

    args = parser.parse_args()

    if not args.input or not args.output:

        raise ValueError('Missing one or more arguments')

    return vars(args)

