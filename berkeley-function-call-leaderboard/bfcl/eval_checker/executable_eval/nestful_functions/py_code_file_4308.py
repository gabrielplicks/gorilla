from typing import AnyStr



def wrap_in_tag(text: AnyStr, wrapper_tag: str = 'wrapper') -> str:

    """Wraps the original text in a new tag.



    The wrapper tag's name is "wrapper" and it is placed around the original text.



    Args:

        text: The text to wrap.

        wrapper_tag: The name of the wrapper tag. Defaults to 'wrapper'.



    Returns:

        A string with the wrapped text.

    """

    return f"<{wrapper_tag}>{text}</{wrapper_tag}>"

