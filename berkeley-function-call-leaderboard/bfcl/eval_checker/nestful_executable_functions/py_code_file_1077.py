from typing import Dict



def parse_apdu(apdu: bytes) -> Dict[str, bytes]:

    """Parses an APDU and returns a dictionary containing the command, length, and payload.



    Args:

        apdu: The binary-encoded APDU.



    Returns:

        A dictionary containing the command, length, and payload.

    """

    command = apdu[:2]

    length = apdu[2:4]

    payload = apdu[4:]

    return {'command': command, 'length': length, 'payload': payload}

