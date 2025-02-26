import numpy as np



def qubit_evolve(state: str, gate: np.ndarray) -> str:

    """

    Applies a quantum gate transformation to a single qubit.



    Args:

        state: The state of the qubit as a string.

        gate: The quantum gate as a 2x2 matrix.

    """

    state_vector = np.array([float(x) for x in state.split(',')])

    state_vector = np.matrix(state_vector).T

    new_state_vector = gate * state_vector

    new_state = ','.join([str(x[0]) for x in new_state_vector.tolist()])



    return new_state

