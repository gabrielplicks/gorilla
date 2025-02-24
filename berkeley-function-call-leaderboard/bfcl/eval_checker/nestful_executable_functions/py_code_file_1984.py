import random

random.seed(42)
from typing import List



def generate_ports() -> List[int]:

    """Generates a list of 500 unique random ports, each being a random integer between 10000 and 60000 inclusive.



    Returns:

        A list of 500 unique random ports.

    """

    generated_ports = set()

    ports = []



    while len(ports) < 500:

        port = random.randrange(10000, 60001)

        if port not in generated_ports:

            generated_ports.add(port)

            ports.append(port)



    return ports

