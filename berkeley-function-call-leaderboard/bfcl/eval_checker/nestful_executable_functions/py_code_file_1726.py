from typing import List



class Message:

    def __init__(self, text: str, date: str):

        self.text = text

        self.date = date



def prepare_message(messages: List[Message]) -> List[str]:

    """Formats a list of messages with their text and date.



    Args:

        messages: A list of Message objects.



    Returns:

        A list of formatted messages.

    """

    formatted_messages = []

    for message in messages:

        text = message.text

        date = message.date

        formatted_message = "Text: {}, Date: {}".format(text, date)

        formatted_messages.append(formatted_message)

    return formatted_messages

