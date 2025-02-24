import math



def generate_lr_sequence(initial_lr: float, num_epochs: int, decay_factor: float) -> list:

    """Generates a sequence of learning rates for a deep learning model.

    The learning rate decays exponentially over the course of `num_epochs` epochs.

    Args:

        initial_lr: The initial learning rate.

        num_epochs: The number of epochs over which to decay the learning rate.

        decay_factor: The factor by which to decay the learning rate each epoch.

    Returns:

        A list of learning rates.

    """

    try:

        lr_sequence = [initial_lr * (decay_factor ** epoch) for epoch in range(num_epochs)]

        return lr_sequence

    except Exception as e:

        print(f"Error generating learning rate sequence: {e}")

        return []

