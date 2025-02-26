from typing import Dict



def initialize_hyperparameters(hyperparams: Dict[str, float]) -> Dict[str, float]:

    """Initializes the model's learning rate, weight decay, and momentum with the values in `hyperparams`.

    If a key is missing, the function uses default values 0.01, 0.01, and 0.9, respectively.

    Args:

        hyperparams: A dictionary containing the hyperparameters.

    """

    default_hyperparams = {

        'learning_rate': 0.01,

        'weight_decay': 0.01,

        'momentum': 0.9

    }

    default_hyperparams.update(hyperparams)  # Update default values with provided hyperparams

    return default_hyperparams  # Return the updated dictionary

