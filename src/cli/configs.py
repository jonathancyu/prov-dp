from argparse import Namespace
from dataclasses import asdict, dataclass


@dataclass()
class Config:
    def merge(self, args: Namespace) -> Namespace:
        args_dict = vars(args)
        args_dict.update(asdict(self))
        return Namespace(**args_dict)


@dataclass
class TreeProcessorConfig(Config):
    epsilon: float
    delta: float
    alpha: float
    beta: float
    gamma: float
    eta: float
    k: int = 250


TREE_CONFIGURATIONS = [
    # All values fixed (Table 4)
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25
    ),
    #
    # Varying privacy budget (Table 4)
    TreeProcessorConfig(
        epsilon=0.1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25
    ),
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25
    ),
    TreeProcessorConfig(
        epsilon=10, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25
    ),
    #
    # Varying hyperparameters (Table 6)
    # Varying delta
    TreeProcessorConfig(
        epsilon=1, delta=0.1, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25
    ),
    # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
    TreeProcessorConfig(
        epsilon=1, delta=0.9, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25
    ),
    #
    # Varying alpha
    TreeProcessorConfig(epsilon=1, delta=0.5, alpha=0.1, beta=0.3, gamma=0.3, eta=0.3),
    # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.9, beta=0.033, gamma=0.033, eta=0.034
    ),
    #
    # Varying beta
    TreeProcessorConfig(epsilon=1, delta=0.5, alpha=0.3, beta=0.1, gamma=0.3, eta=0.3),
    # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.033, beta=0.9, gamma=0.033, eta=0.034
    ),
    #
    # Varying gamma
    TreeProcessorConfig(epsilon=1, delta=0.5, alpha=0.3, beta=0.3, gamma=0.1, eta=0.3),
    # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.033, beta=0.033, gamma=0.9, eta=0.034
    ),
    #
    # Varying Eta
    TreeProcessorConfig(epsilon=1, delta=0.5, alpha=0.3, beta=0.3, gamma=0.3, eta=0.1),
    # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.033, beta=0.033, gamma=0.034, eta=0.9
    ),
]

K_CONFIGURATIONS = [
    # Varying k (probably need to add it above XD)
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25, k=5
    ),
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25, k=250
    ),
    TreeProcessorConfig(
        epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25, k=500
    ),
]
