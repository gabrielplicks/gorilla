import json

from typing import Any



def handle_json_rpc_request(request: str) -> str:

    """Handles JSON-RPC requests for a server.



    Args:

        request: A JSON-RPC request string.



    Returns:

        A JSON string containing the result of the request.

    """

    request_dict = json.loads(request)

    method_name = request_dict.get("method")

    if method_name not in globals():

        raise ValueError("Invalid method name")

    params = request_dict.get("params", [])

    if not isinstance(params, list):

        raise ValueError("Invalid parameters")

    id = request_dict.get("id")

    result = globals()[method_name](*params)

    return json.dumps({"id": id, "result": result})

