from typing import BinaryIO



def extract_packets(packet_ids: list[int], file_path: str) -> list:

    """Extracts packets from a binary file based on a list of packet IDs.



    Args:

        packet_ids: A list of packet IDs to extract from the binary file.

        file_path: The path to the binary file.



    Returns:

        A list of packet data corresponding to the provided packet IDs.

    """

    with open(file_path, 'rb') as f:

        packets = ORB_reap_packets(f, packet_ids)

    return packets

