def create_prompt(name: str, prompt: str) -> str:

    """Creates a prompt string by concatenating the name and prompt, adding a colon after the name, and adding a space between the name and prompt.



    Args:

        name: The name of a person.

        prompt: A message to be displayed to the user.



    Raises:

        ValueError: If either name or prompt is empty or not a string.

    """

    if not isinstance(name, str) or not name:

        raise ValueError("name must be a non-empty string")

    if not isinstance(prompt, str) or not prompt:

        raise ValueError("prompt must be a non-empty string")



    return f"{name}: {prompt}"

