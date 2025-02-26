from typing import List



def get_cpu_usage_less_than_10(file_path: str) -> List[float]:

    """Returns a list of CPU usage values that are less than 10%.



    Args:

        file_path: The path to the file containing CPU usage values.



    Returns:

        A list of CPU usage values that are less than 10%.

    """

    cpu_usage_values = []

    try:

        with open(file_path, 'r') as file:

            for line in file:

                timestamp, cpu_usage = line.split()

                if float(cpu_usage) < 10:

                    cpu_usage_values.append(float(cpu_usage))

    except FileNotFoundError:

        pass

    return cpu_usage_values

