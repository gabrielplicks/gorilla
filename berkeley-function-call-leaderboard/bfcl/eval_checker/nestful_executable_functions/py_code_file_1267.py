from typing import Union



def create_text_message(sender_id: Union[int, str], receiver_id: Union[int, str]) -> str:

    """Creates a text message from the sender to the receiver.

    Args:

        sender_id: The ID of the sender.

        receiver_id: The ID of the receiver.

    """

    message_text = f"From sender_id {sender_id} to receiver_id {receiver_id}"

    return message_text

