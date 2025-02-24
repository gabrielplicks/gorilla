import random

random.seed(42)
import typing



def construct_service_name(cluster_name: str, service_id: str) -> str:

    """Constructs a service name given a cluster_name and service_id.

    The function returns a string with the format cluster_name.service_id.instance_id,

    where instance_id is a random integer between 0 and 2048.

    Args:

        cluster_name: The name of the cluster.

        service_id: The id of the service.

    """

    instance_id = random.randint(0, 2048)

    return f"{cluster_name}.{service_id}.{instance_id}"

