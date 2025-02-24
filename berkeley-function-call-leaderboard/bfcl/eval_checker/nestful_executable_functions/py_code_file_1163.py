from typing import List, Dict



def get_reward_values(rewards: List[Dict[str, str]]) -> List[int]:

    """Extracts the reward values for the "money" reward type from a list of dictionaries.



    Args:

        rewards: A list of dictionaries representing rewards. Each dictionary contains the keys "name", "reward_type", and "value".



    Returns:

        A list of reward values for the "money" reward type.

    """

    return [reward["value"] for reward in rewards if reward["reward_type"] == "money"]

