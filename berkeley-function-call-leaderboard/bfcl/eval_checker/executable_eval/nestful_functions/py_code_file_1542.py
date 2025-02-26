def compose_response(message: str) -> str:

    """Composes an HTTP response body in JSON format.



    Args:

        message: The message to be included in the JSON object.



    Returns:

        The HTTP response body in JSON format.

    """

    json_string = compose_json(message)

    http_response = compose_http_response(json_string)

    return http_response

