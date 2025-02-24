from typing import Union



class CPU:

    """A CPU device."""



class GPU:

    """A GPU device with a number of cores and memory size."""



    def __init__(self, num_cores: int, memory_size: int):

        """Initializes a GPU device with a number of cores and memory size.

        Args:

            num_cores: The number of cores in the GPU.

            memory_size: The memory size of the GPU.

        """

        self.num_cores = num_cores

        self.memory_size = memory_size



def my_func(device: Union[CPU, GPU]) -> str:

    """Returns a string representation of the given device.

    Args:

        device: A CPU or GPU device.

    """

    if isinstance(device, CPU):

        return f"device: {device}"

    else:

        return f"device: {device}, num_cores: {device.num_cores}, memory_size: {device.memory_size}"

