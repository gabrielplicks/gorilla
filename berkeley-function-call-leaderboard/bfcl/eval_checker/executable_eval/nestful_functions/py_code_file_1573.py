from typing import Tuple



def replace_ip_in_domain(ip: str, domain: str) -> str:

    """Replaces the first component of a domain name with a new IP address.



    Args:

        ip: The new IP address.

        domain: The original domain name.



    Returns:

        The new domain name with the IP address replaced.

    """

    components = domain.split('.')

    components[0] = ip

    return '.'.join(components)

