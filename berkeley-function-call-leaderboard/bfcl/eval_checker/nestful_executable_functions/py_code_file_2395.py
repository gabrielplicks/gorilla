import base64



def generate_authenticate_command(username: str, password: str) -> str:

    """Generates an AUTHENTICATE command for SASL PLAIN mechanism.



    Args:

        username: The username to be base64-encoded.

        password: The password to be base64-encoded.

    """

    encoded_username = base64.b64encode(username.encode('utf-8')).decode('utf-8')

    encoded_password = base64.b64encode(password.encode('utf-8')).decode('utf-8')

    command = 'AUTHENTICATE PLAIN 00' + encoded_username + '00' + encoded_password



    return command

