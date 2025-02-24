import json

from typing import Any



def get_concatenated_text(json_file_path: str) -> str:

    """Retrieves the text from a JSON file, removes all line breaks, and returns a concatenated string.



    Args:

        json_file_path: The path to the JSON file.



    Returns:

        A string containing the concatenated text with all line breaks removed.

    """

    with open(json_file_path, 'r') as f:

        data: Any = json.load(f)

    text_list: list[str] = data["text"]

    concatenated_text: str = "".join(text_list)

    concatenated_text = concatenated_text.replace("\n", "")

    return concatenated_text

