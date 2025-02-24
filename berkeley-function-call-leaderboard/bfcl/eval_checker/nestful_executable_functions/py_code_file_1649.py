from typing import Tuple, Dict



def convert_exception_to_dict(exception_tuple: Tuple[type, str]) -> Dict[str, str]:

    """Converts an exception tuple to a dictionary.

    Args:

        exception_tuple: A tuple containing an exception type and a message.

    Returns:

        A dictionary with the exception name as the key and the message as the value.

    """

    if not isinstance(exception_tuple, tuple):

        return {}  # Return an empty dictionary if the input is not a tuple



    exception_name = exception_tuple[0].__name__

    exception_message = exception_tuple[1]



    return {exception_name: exception_message}

