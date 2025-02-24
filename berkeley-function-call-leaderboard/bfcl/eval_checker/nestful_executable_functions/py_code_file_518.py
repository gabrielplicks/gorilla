import json



def build_response(status: str, data: dict) -> dict:

    """Builds a response dictionary containing the provided status and data encoded as a JSON string.



    Args:

        status: The status of the response.

        data: The data to be encoded as a JSON string.

    """

    return {

        'status': status,

        'data': json.dumps(data)

    }

