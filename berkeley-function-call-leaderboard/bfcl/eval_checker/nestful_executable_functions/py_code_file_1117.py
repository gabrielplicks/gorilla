from typing import List



def shape_rewards(rewards: List[float]) -> List[float]:

    """Shapes rewards in reinforcement learning by subtracting the mean reward from each reward.



    Args:

        rewards: A list of rewards.



    Raises:

        ValueError: If the mean reward is negative.



    Returns:

        A list of shaped rewards.

    """

    mean_reward = sum(rewards) / len(rewards)

    if mean_reward < 0:

        raise ValueError("Mean reward is negative")

    return [reward - mean_reward for reward in rewards]

