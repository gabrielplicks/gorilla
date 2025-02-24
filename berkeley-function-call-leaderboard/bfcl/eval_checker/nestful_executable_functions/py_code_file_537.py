from typing import List



def inequality(weights: List[float]) -> float:

    """Calculates the inequality coefficient with the given weight coefficient W.

    The inequality coefficient is calculated as follows:

    inequality(W) = sum_i(W_i * (1 - W_i))

    where W_i is the weight of the i-th class.

    Args:

        weights: A list of floats representing the class weights.

    """

    return sum(weight * (1 - weight) for weight in weights)

