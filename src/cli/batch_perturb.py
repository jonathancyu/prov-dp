from dataclasses import dataclass
from copy import deepcopy

from .perturb import run_processor, parse_args


@dataclass
class Configuration:
    epsilon: float
    delta: float
    alpha: float
    beta: float
    gamma: float
    eta: float


def batch_run(args):

    configurations = [
        # All values fixed (Table 4)
        Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5, eta=0.5),
        # Varying privacy budget (Table 4)
        Configuration(epsilon=0.1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5, eta=0.5),
        Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5, eta=0.5),
        Configuration(epsilon=10, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5, eta=0.5),
        # Varying hyperparameters (Table 6)
        # Varying delta
        Configuration(epsilon=1, delta=0.1, alpha=0.5, beta=0.5, gamma=0.5, eta=0.5),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Configuration(epsilon=1, delta=0.9, alpha=0.5, beta=0.5, gamma=0.5, eta=0.5),
        # Varying alpha
        Configuration(epsilon=1, delta=0.5, alpha=0.1, beta=0.5, gamma=0.5, eta=0.5),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Configuration(epsilon=1, delta=0.5, alpha=0.9, beta=0.5, gamma=0.5, eta=0.5),
        # Varying beta
        Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.1, gamma=0.5, eta=0.5),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.9, gamma=0.5, eta=0.5),
        # Varying gamma
        Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.1, eta=0.5),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.9, eta=0.5),
    ]

    for config in configurations:
        current_args = deepcopy(args)
        current_args.epsilon = config.epsilon
        current_args.delta = config.delta
        current_args.alpha = config.alpha
        current_args.beta = config.beta
        current_args.gamma = config.gamma
        current_args.eta = config.eta
        run_processor(current_args)
        print()
        print()


def main(args):
    batch_run(args)


if __name__ == "__main__":
    main(parse_args())
