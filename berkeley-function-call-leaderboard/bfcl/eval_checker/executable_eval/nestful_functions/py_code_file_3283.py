from typing import Any, Dict



def extract_html_from_json(data: Dict[str, Any]) -> str:

    """Extracts the rendered HTML from the `template` property inside a JSON object.



    Args:

        data: The input JSON object.



    Returns:

        The extracted HTML content as a string.

    """

    return data["template"]

