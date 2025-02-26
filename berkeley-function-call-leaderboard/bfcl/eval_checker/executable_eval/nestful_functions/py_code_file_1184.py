from typing import Tuple



class Agent:

    def perform_action(self, env):

        pass



    def receive_reward(self, reward: float, success: bool):

        pass



class Environment:

    def perform_action(self, action):

        pass



def simulate_rl_env(agent: Agent, env: Environment) -> Tuple[str, float, bool]:

    """Simulates a reinforcement learning environment where an agent interacts with the environment and receives outputs.



    Args:

        agent: The agent that performs an action within the environment.

        env: The environment where the agent performs the action.



    Returns:

        The action performed by the agent, the reward received, and a Boolean value indicating whether the action was successful.

    """

    action = agent.perform_action(env)

    reward, success = env.perform_action(action)

    agent.receive_reward(reward, success)



    return action, reward, success

