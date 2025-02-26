from typing import List, Tuple

import random

random.seed(42)


def gen_agent_pairings(agents: List[str]) -> List[Tuple[str, str]]:

    """Generates pairs of agents for game playing, preventing the same agent from playing against itself.

    If the number of agents is odd, a single agent will be left unpaired.

    Args:

        agents: A list of agents.

    """

    random.shuffle(agents)

    pairings = []

    for i in range(0, len(agents), 2):

        if i + 1 < len(agents):

            pairings.append((agents[i], agents[i + 1]))

        else:

            pairings.append((agents[i],))

    return pairings

