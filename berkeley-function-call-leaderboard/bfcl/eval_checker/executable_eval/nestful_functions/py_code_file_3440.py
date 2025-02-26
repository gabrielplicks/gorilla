from typing import List



def encode_tags(tags: List[object]) -> str:

    """Encodes a list of tags into a URL-friendly string separated by dashes (-).

    Args:

        tags: A list of tags, each with a `.name` attribute.

    """

    return "-".join(tag.name.lower() for tag in tags)

