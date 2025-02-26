import base64



def encode_dns_name(name_and_domain: str) -> str:

    """Encodes the given name and domain using base64 encoding.



    Args:

        name_and_domain: The name and domain to be encoded.



    """

    return base64.b64encode(name_and_domain.encode('utf-8')).decode('utf-8')

