from typing import List, Dict



def calculate_total_duration(episodes: List[Dict[str, int]]) -> int:

    """Calculates the total duration of a list of episodes for a television show.

    Args:

        episodes: A list of dictionaries, where each dictionary represents an episode and has a "length" key.

    """

    durations = [episode["length"] for episode in episodes]

    total_duration = sum(durations)

    return total_duration

