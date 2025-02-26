from typing import List



def count_state_changes(states: List[str]) -> int:

    """Counts the number of state changes in a system represented by a list of strings.



    Args:

        states: A list of strings representing the state changes in the system.



    Raises:

        TypeError: If `states` is not a list of strings.

    """

    if not isinstance(states, list) or not all(isinstance(state, str) for state in states):

        raise TypeError("states must be a list of strings")



    num_changes = 0

    previous_state = None

    for state in states:

        if previous_state and state != previous_state:

            num_changes += 1

        previous_state = state



    return num_changes

