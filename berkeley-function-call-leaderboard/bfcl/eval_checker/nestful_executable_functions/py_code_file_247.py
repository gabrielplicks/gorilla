from typing import Dict, List



def validate_content_type(field: Dict[str, str], allowed_types: List[str]) -> bool:

    """Validates that a given HTML form field has a valid content type based on a given list of allowed content types.



    Args:

        field: A dictionary containing the HTML form field.

        allowed_types: A list of allowed content types.

    """

    return field['content_type'] in allowed_types

