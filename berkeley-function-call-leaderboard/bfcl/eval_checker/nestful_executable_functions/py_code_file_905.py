import re



def has_link_in_description(post: Any) -> bool:

    """Determines if the `description` attribute of a given `post` object contains the word "link" (case insensitive).



    Args:

        post: The post object.



    Returns:

        True if the `description` attribute contains the word "link", False otherwise.

    """

    return bool(re.search(r"\blink\b", post.description, flags=re.IGNORECASE))

