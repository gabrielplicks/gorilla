from typing import List



def find_rolling_mean(data: List[float], window: int) -> List[float]:

    """Calculates the rolling mean of a given list of numbers.



    Args:

        data: A list of numbers.

        window: An integer representing the size of the window for calculating the rolling mean.



    Returns:

        A list of rolling means, where each element in the returned list is the mean of the

        corresponding window in the input list. The final result is rounded to 4 decimal places.

    """

    rolling_means = []



    for i in range(len(data) - window + 1):

        window_values = data[i:i+window]

        mean = sum(window_values) / len(window_values)

        mean = round(mean, 4)  # Round to 4 decimal places

        rolling_means.append(mean)



    return rolling_means

