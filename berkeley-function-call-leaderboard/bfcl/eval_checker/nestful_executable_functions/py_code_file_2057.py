from typing import Union



def format_alert_message(message: str) -> Union[str, None]:

    """Formats an alert message with an appropriate Bootstrap color class.



    Args:

        message: The message string to format.



    Returns:

        The formatted HTML string for a Bootstrap alert, or None if the message is empty.

    """

    if not message:

        return None



    color_class = "danger"



    if message.isalpha():

        color_class = "success"

    elif message.isdigit():

        color_class = "info"

    elif message.isalnum():

        color_class = "warning"



    return f'<div class="alert alert-{color_class}">{message}</div>'

