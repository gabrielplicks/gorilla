from typing import AnyStr



def read_string(prompt: AnyStr) -> AnyStr:

    """

    Prompts the user for an input string.

    Handles the following exceptions:

    1. KeyboardInterrupt: The user pressed `ctrl+c` to interrupt the program.

       In this case, the function asks the user to type a string manually, as input, and returns that string.

    2. EOFError: The user entered `ctrl+d` to send an end-of-file signal.

       In this case, the function prints a message and returns an empty string.



    Args:

        prompt: The prompt to display to the user.



    Returns:

        The input string.

    """

    try:

        input_string = input(prompt)

    except KeyboardInterrupt:

        input_string = input('Please enter a string manually: ')

    except EOFError:

        print('End-of-file signal received. Returning an empty string.')

        input_string = ''



    return input_string

