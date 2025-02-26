from typing import List, Dict



def arg_parser(args: List[str], options: Dict[str, str]) -> Dict[str, str]:

    """Parses command line arguments and stores them in a dictionary.



    Args:

        args: A list of command line arguments.

        options: An empty dictionary to store the parsed arguments.



    Returns:

        The dictionary with the parsed arguments.

    """

    for i in range(len(args)):

        if args[i] == '-x':

            options['x'] = int(args[i + 1])

        elif args[i] == '-y':

            options['y'] = int(args[i + 1])

        elif args[i] == '-n':

            options['name'] = args[i + 1]

        elif args[i] == '-m':

            options['message'] = ' '.join(args[i + 1:])

    return options

