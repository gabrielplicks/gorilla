from datetime import datetime



def print_error(message: str) -> str:

    """Prints a nicely-formatted error message to the user, outputting a string consisting of a message and a timestamp.



    The message is in the format: 'ERROR: {timestamp}: {message}'. The timestamp is formatted as '%Y-%m-%d %H:%M:%S' using Python's datetime module, and the message is colored in bright red using ANSI escape executable_functions.



    Args:

        message: The error message to be displayed.



    Returns:

        The formatted error message string.

    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    error_message = f'\033[91mERROR: {timestamp}: {message}\033[0m'

    print(error_message)

    return error_message

