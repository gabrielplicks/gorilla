from typing import Dict, List



def track_update_durations(batch_size: int, time_interval: int, update_durations: List[int]) -> Dict[int, List[int]]:

    """

    Tracks the update durations of multiple batches over time.



    Args:

        batch_size: The size of the batch.

        time_interval: The time interval for calculating the average update duration.

        update_durations: A list of update durations.



    Returns:

        A dictionary containing the batch sizes as keys and their corresponding update durations as values.

    """

    updates_per_batch = {batch_size: []}



    for duration in update_durations:

        if duration % time_interval == 0:

            updates_per_batch[batch_size].append(duration)



    return updates_per_batch

