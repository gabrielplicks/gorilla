import random

random.seed(42)
from typing import Any



def convert_config_to_string(config: Any) -> str:

    """Converts a config object into a string with the format `config_id_max_batch_size_unique_id`.



    Args:

        config: The config object with `config_id` and `max_batch_size` attributes.



    Returns:

        A string in the format `config_id_max_batch_size_unique_id`.

    """

    unique_id = random.randint(10 ** 9, 10 ** 10 - 1)

    output_string = "_".join([config.config_id, str(config.max_batch_size), str(unique_id)])



    return output_string

