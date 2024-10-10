from dataclasses import dataclass
from copy import deepcopy
from math import gamma

from numpy.random import beta

from .perturb import run_processor, parse_args


@dataclass
class Config:
    epsilon: float
    delta: float
    alpha: float
    beta: float
    gamma: float
    eta: float


def batch_run(args):

    configurations = [
        # All values fixed (Table 4)
        Config(epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25),
        #
        # Varying privacy budget (Table 4)
        Config(epsilon=0.1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25),
        Config(epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25),
        Config(epsilon=10, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25),
        #
        # Varying hyperparameters (Table 6)
        # Varying delta
        Config(epsilon=1, delta=0.1, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Config(epsilon=1, delta=0.9, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25),
        #
        # Varying alpha
        Config(epsilon=1, delta=0.5, alpha=0.1, beta=0.3, gamma=0.3, eta=0.3),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Config(epsilon=1, delta=0.5, alpha=0.9, beta=0.033, gamma=0.033, eta=0.034),
        #
        # Varying beta
        Config(epsilon=1, delta=0.5, alpha=0.3, beta=0.1, gamma=0.3, eta=0.3),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Config(epsilon=1, delta=0.5, alpha=0.033, beta=0.9, gamma=0.033, eta=0.034),
        #
        # Varying gamma
        Config(epsilon=1, delta=0.5, alpha=0.3, beta=0.3, gamma=0.1, eta=0.3),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Config(epsilon=1, delta=0.5, alpha=0.033, beta=0.033, gamma=0.9, eta=0.034),
        #
        # Varying Eta
        Config(epsilon=1, delta=0.5, alpha=0.3, beta=0.3, gamma=0.3, eta=0.1),
        # Configuration(epsilon=1, delta=0.5, alpha=0.5, beta=0.5, gamma=0.5),
        Config(epsilon=1, delta=0.5, alpha=0.033, beta=0.033, gamma=0.034, eta=0.9),
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
