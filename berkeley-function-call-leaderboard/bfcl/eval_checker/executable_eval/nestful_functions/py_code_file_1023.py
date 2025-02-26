from typing import Dict, List



def request_allowed(request: Dict, allowed_methods: List[str]) -> bool:

    """Checks whether the request method is in the allowed methods list.



    Args:

        request: The request object.

        allowed_methods: The list of allowed methods.



    Returns:

        True if the request method is in the allowed methods list, False otherwise.

    """

    return isinstance(request, dict) and request.get("method") in allowed_methods

