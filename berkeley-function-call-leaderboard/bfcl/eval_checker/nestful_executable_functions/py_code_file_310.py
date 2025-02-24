from typing import AnyStr



def replace_route_with_path(error_message: AnyStr) -> AnyStr:

    """Replaces the substring "route" with "path" in the given error message.



    Args:

        error_message: The error message to modify.



    Returns:

        A new error message with "route" replaced with "path".

    """

    return error_message.replace("route", "path")

