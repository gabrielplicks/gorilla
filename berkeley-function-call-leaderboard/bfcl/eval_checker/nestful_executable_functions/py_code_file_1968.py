from collections import deque

from typing import List



def moving_average_sliding_window(stream: List[float], window_size: int) -> None:

    """Calculates the moving average of a sliding window over a stream of numbers.



    Args:

        stream: A list of numbers representing the stream.

        window_size: The size of the sliding window.

    """

    queue = deque(maxlen=window_size)

    average = 0

    for num in stream:

        queue.append(num)

        average = sum(queue) / float(len(queue))

        print(f"Average: {average}")

