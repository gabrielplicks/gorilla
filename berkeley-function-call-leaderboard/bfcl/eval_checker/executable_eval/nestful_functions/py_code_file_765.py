from typing import List



def custom_lr_scheduler(epoch: int, num_epochs: int) -> float:

    """

    Calculates the learning rate for a given epoch based on a custom learning rate scheduler.



    The learning rate scheduler decreases the learning rate linearly from its initial value at the beginning of each epoch to zero at the end of the epoch.



    Args:

        epoch: The current epoch number.

        num_epochs: The total number of epochs for the training process.



    Returns:

        The learning rate for the corresponding epoch.

    """

    initial_lr = 0.1

    lr_list = [initial_lr * (1 - (epoch / num_epochs)) for epoch in range(num_epochs)]

    return lr_list[epoch]

