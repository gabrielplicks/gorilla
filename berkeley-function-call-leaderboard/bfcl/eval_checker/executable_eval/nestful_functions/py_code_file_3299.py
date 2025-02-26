from typing import Any



def process_request(request: Any) -> str:

    """

    Formats a string with a request's text if it exists, otherwise returns

    a default string.

    Args:

        request: A request object that may contain multiple fields.

    """

    if hasattr(request, 'text'):

        return f"Received the following text: {request.text}"

    else:

        return "No text received"

