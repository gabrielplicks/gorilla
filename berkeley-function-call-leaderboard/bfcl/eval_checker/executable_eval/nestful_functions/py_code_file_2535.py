from typing import Dict



def convert_tags(tags_dict: Dict[str, str]) -> list[str]:

    """Converts a dictionary of tag names and corresponding tag values into a list of tags in the format "tag name:tag value".



    Args:

        tags_dict: A dictionary of tag names and corresponding tag values.



    Returns:

        A list of tags in the format "tag name:tag value".

    """

    return [f"{key}:{value}" for key, value in tags_dict.items()]

