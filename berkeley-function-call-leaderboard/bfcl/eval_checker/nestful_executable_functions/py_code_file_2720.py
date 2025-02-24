import re

from typing import Tuple



def get_port_name_and_interface_type(flow_interface_name: str) -> Tuple[str, str]:

    """

    Returns the port name and interface type of a flow interface name.



    Args:

        flow_interface_name: The flow interface name.



    Returns:

        A tuple containing the port name and interface type.



    Raises:

        ValueError: If the flow interface name doesn't match the pattern.

    """

    pattern = r"(?P<port_name>[A-Za-z]+)(?P<interface_type>\d\/\d)"

    match = re.search(pattern, flow_interface_name)

    if match:

        port_name = match.group("port_name")

        interface_type = match.group("interface_type")

        return (port_name, interface_type)

    else:

        raise ValueError("Invalid flow interface name.")

