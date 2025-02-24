from typing import List



def format_log_message(message: str, words: List[str]) -> str:

    """Formats a log message with a list of words.

    Args:

        message: The original message.

        words: A list of words to insert into the message.

    """

    formatted_words = ["\n" + word for word in words]

    return message + "".join(formatted_words)

