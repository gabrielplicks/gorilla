from typing import Union



def resolve_port_for_app_protocol(app_protocol: str) -> Union[int, None]:

    """Resolves the port number for a given application protocol.



    Args:

        app_protocol: The application protocol string.



    Returns:

        The corresponding port number as an integer, or None if the protocol is not recognized.

    """

    if app_protocol == 'HTTP':

        return 80

    elif app_protocol == 'HTTPS':

        return 443

    elif app_protocol == 'FTP':

        return 21

    elif app_protocol in ['TCP', 'UDP', 'POP3', 'IMAP', 'SMTP']:

        return 53

    else:

        return None

