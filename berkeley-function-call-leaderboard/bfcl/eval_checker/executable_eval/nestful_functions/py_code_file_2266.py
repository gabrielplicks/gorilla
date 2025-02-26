from typing import List



def detect_event(state_transitions: List[int]) -> bool:

    """Detects an event in a list of state transitions.



    An event is defined as a state transition from state 0 to state 1 and back to state 0.



    Args:

        state_transitions: A list of state transitions.



    Returns:

        True if an event is detected, False otherwise.

    """

    for i in range(len(state_transitions) - 2):

        if state_transitions[i] == 0 and state_transitions[i+1] == 1 and state_transitions[i+2] == 0:

            return True

    return False

