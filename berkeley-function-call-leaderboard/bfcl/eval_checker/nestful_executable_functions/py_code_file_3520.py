from typing import List, Tuple



def detect_overlapping_time_slot(time_slots: List[Tuple[int, int]]) -> bool:

    """Detects whether there is any overlapping time slot among the input list.



    Args:

        time_slots: A list of time slots in the format `[(start_time1, end_time1), (start_time2, end_time2), ...]`.



    Returns:

        True if there is any overlapping time slot, False otherwise.

    """

    sorted_time_slots = sorted(time_slots, key=lambda x: x[0])

    for i in range(len(sorted_time_slots) - 1):

        if sorted_time_slots[i][1] >= sorted_time_slots[i + 1][0]:

            return True



    return False

