from typing import List, Dict



def filter_content_types(content_types: List[str], schema: Dict[str, Dict]) -> List[str]:

    """

    Filters a list of content types to only include those that have a schema defined.



    Args:

        content_types: A list of content types.

        schema: A dictionary mapping content types to their schemas.



    Returns:

        A list of content types that have a schema defined.

    """

    return [content_type for content_type in content_types if content_type in schema]

