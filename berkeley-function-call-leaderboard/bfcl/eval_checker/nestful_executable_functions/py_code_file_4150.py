from typing import Literal



def ttl_computation(packet: object) -> int:

    """Computes time-to-live (TTL) for a given packet.



    Args:

        packet: The packet for which TTL is to be computed.



    Returns:

        The computed TTL as an integer.

    """

    ttl_map = {

        'TCP': 64 if not getattr(packet, 'mss', None) else 64,

        'UDP': 64 if not getattr(packet, 'dscp', None) else 64 - getattr(packet, 'dscp'),

        'ICMP': 64,

    }

    return ttl_map[packet.type]

