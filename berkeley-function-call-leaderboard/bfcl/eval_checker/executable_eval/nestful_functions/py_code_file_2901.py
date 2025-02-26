from typing import Union



def probability_of_decay(half_life: Union[int, float], time: Union[int, float]) -> float:

    """Calculates the probability of a particle decaying in a given time.



    Args:

        half_life: The half-life of the radioactive substance.

        time: The time elapsed.

    """

    return 1 - 2 ** (-time / half_life)



def probability_of_n_particles_decay(n: int, half_life: Union[int, float], time: Union[int, float]) -> float:

    """Calculates the probability of a given number of particles decaying in a given time.



    Args:

        n: The number of particles.

        half_life: The half-life of the radioactive substance.

        time: The time elapsed.

    """

    probability = 1

    for _ in range(n):

        probability *= probability_of_decay(half_life, time)

    return probability

