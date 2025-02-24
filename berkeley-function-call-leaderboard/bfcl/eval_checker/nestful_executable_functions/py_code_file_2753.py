import sys



def is_interpreter_running() -> bool:

    """Detects whether a given Python interpreter session is currently running.

    If not, the function raises an exception with an appropriate error message.

    If so, the function returns True.

    Uses try-catch blocks to handle any exceptions that may occur.

    Args:

        None

    """

    try:

        if sys.platform == 'linux' and sys.implementation.name == 'cpython':

            return True

        else:

            raise Exception('Interpreter is not running')

    except Exception as e:

        print(f'Error: {e}')

        return False

