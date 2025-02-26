from typing import Any



def mock_server(request: Any) -> str:

    """Mocks a server response to a GET request.

    Args:

        request: The request object.

    Returns:

        The mocked server response.

    """

    assert request.method == 'GET'

    return """HTTP/1.1 200 OK

Content-Type: application/json



{

    "success": true,

    "data": {

        "message": "Hello, world!"

    }

}"""

