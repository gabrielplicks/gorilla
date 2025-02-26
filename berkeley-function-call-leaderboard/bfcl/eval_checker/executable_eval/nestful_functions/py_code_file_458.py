import math



def log_prob(probabilities: list[float]) -> float:

    """Calculates the log probability of each distribution.

    Args:

        probabilities: The list of probabilities.

    """

    log_probs = [math.log(p) for p in probabilities]

    return sum(log_probs)

