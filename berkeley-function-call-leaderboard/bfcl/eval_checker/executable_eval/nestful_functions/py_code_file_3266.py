import re



def remove_autopersonalization_from_message(message: str) -> str:

    """Removes the substring 'I am will' from a given message.



    Args:

        message: The input message.



    Returns:

        The cleaned message.

    """

    assert isinstance(message, str), "Input must be a string"

    assert "I am will" in message, "Input must contain 'I am will'"



    normalized_message = message.strip()

    cleaned_message = re.sub(r"(?i)I am will", "", normalized_message)

    return cleaned_message.strip()

