from typing import Union



def reward_function(is_agent_alive: Union[bool, int]) -> int:

    """Calculates the reward for an agent that can live or die.



    Args:

        is_agent_alive: The life state of the agent.



    Returns:

        The reward for the agent.

    """

    if is_agent_alive:

        return 1

    else:

        return -100

