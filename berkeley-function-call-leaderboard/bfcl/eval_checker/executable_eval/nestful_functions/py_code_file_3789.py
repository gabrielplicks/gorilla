import numpy as np



def softmax_action_selector(logits: np.ndarray, temperature: float) -> int:

    """Selects an action based on the softmax distribution of the logits values.



    Args:

        logits: A 1D array or list of numbers representing the logits.

        temperature: A scalar representing the temperature.



    Returns:

        An index representing the "chosen" action out of all of the possible actions.

    """

    softmax_distribution = np.exp(logits) / np.sum(np.exp(logits))

    adjusted_distribution = np.power(softmax_distribution, 1 / temperature)

    adjusted_distribution /= np.sum(adjusted_distribution)

    action = np.random.choice(len(logits), p=adjusted_distribution)



    return action

