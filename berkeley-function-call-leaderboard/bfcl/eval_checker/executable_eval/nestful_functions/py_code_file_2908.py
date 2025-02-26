from typing import Optional



def generate_comment(name: str, comment: Optional[str] = "", verbose: Optional[bool] = True) -> Optional[str]:

    """Generates a comment in the specified format.



    Args:

        name: The name to include in the comment.

        comment: The comment to include in the comment block. Defaults to an empty string.

        verbose: A flag indicating whether to include the comment. Defaults to True.



    Raises:

        ValueError: If the name parameter is missing or invalid.

    """

    if not name:

        raise ValueError("The name parameter is required.")

    comment_block = f"""

Author: {name}

Comment: {comment}

"""

    return comment_block if verbose else ""

