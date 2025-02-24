from typing import Optional



def build_greeting_message(name: str, title: Optional[str] = None) -> str:

    """

    Builds a greeting message with the provided name and title.



    Args:

        name: The name of the person to greet.

        title: The title of the person to greet. If not provided, the default greeting message is used.



    Returns:

        A greeting message that includes the provided name and title.

    """

    if title:

        greeting = f"Hello, {title} {name}!"

    else:

        greeting = f"Hello, {name}!"



    return greeting

