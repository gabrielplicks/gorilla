import numpy as np



def logsoftmax(input: np.ndarray) -> np.ndarray:

    """Calculates the logsoftmax of a given input.



    Args:

        input: A numpy array of numeric values.



    Returns:

        A numpy array of logsoftmax values corresponding to each input value.

    """

    exp_input = np.exp(input)

    sum_exp_input = np.sum(exp_input)

    softmax_output = exp_input / sum_exp_input

    logsoftmax_output = np.log(softmax_output)



    return logsoftmax_output

