from typing import Dict



def modify_http_header(header: Dict[str, str]) -> Dict[str, str]:

    """Modifies the `Access-Control-Allow-Origin` HTTP header in a server response to allow cross-origin resource sharing (CORS).



    Args:

        header: The HTTP header to modify.



    Returns:

        The modified HTTP header as a dictionary.

    """

    header_dict = dict(header)

    header_dict["Access-Control-Allow-Origin"] = "*"

    return header_dict

