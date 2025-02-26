from typing import Any



def parse_and_format_exception(exception: Any) -> str:

    """Formats an exception as a string in the format "Python Exception: <exception_name> (<exception_message>)".



    Args:

        exception: The exception to format.



    Returns:

        A string formatted as "Python Exception: <exception_name> (<exception_message>)".

    """

    try:

        exception_name = exception.__class__.__name__

        exception_message = str(exception)

        formatted_exception = f'Python Exception: {exception_name} ({exception_message})'

        return formatted_exception

    except Exception as e:

        return f'Python Exception: Unknown'

