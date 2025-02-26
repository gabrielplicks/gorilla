import json

from typing import Any, Dict, List



def parse_json_to_string(json_string: str) -> str:

    """Parses a JSON string into a list of dictionaries, where each dictionary contains the "id", "title", and "body" fields.

    Formats the output as a string with the following format:



    ID: {id}

    Title: {title}

    Body: {body}



    ID: {id}

    Title: {title}

    Body: {body}



    Args:

        json_string: The JSON string to parse.



    Returns:

        The formatted string.

    """

    data: List[Dict[str, Any]] = json.loads(json_string)

    output: str = ""

    for item in data:

        output += f"ID: {item['id']}\nTitle: {item['title']}\nBody: {item['body']}\n\n"

    return output

