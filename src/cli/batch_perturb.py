from dataclasses import dataclass
import itertools
from copy import deepcopy

from .perturb import run_processor, parse_args


@dataclass
class Configuration:
    epsilon: float
    delta: float
    alpha: float


def batch_run(args):

    configurations = [
        # All values fixed (Table 4)
        Configuration(epsilon=1, delta=0.5, alpha=0.5),
        # Varying privacy budget (Table 4)
        Configuration(epsilon=0.1, delta=0.5, alpha=0.5),
        Configuration(epsilon=1, delta=0.5, alpha=0.5),
        Configuration(epsilon=10, delta=0.5, alpha=0.5),
        # Varying hyperparameters (Table 6)
        # Varying alpha
        Configuration(epsilon=1, delta=0.5, alpha=0.1),
        Configuration(epsilon=1, delta=0.5, alpha=0.5),
        Configuration(epsilon=1, delta=0.5, alpha=0.9),
        # Varying delta
        Configuration(epsilon=1, delta=0.1, alpha=0.5),
        Configuration(epsilon=1, delta=0.5, alpha=0.5),
        Configuration(epsilon=1, delta=0.9, alpha=0.5),
    ]

    for config in configurations:
        epsilon_1 = config.epsilon * config.delta
        epsilon_2 = config.epsilon * (1 - config.delta)
        alpha = config.alpha
        beta = 0
        gamma = 0

        current_args = deepcopy(args)
        print(
            f"(0) beginning epsilon_1={epsilon_1}, alpha={alpha}, beta={beta}, gamma={gamma}"
        )
        current_args.epsilon_1 = epsilon_1
        current_args.epsilon_2 = epsilon_2
        current_args.alpha = alpha
        current_args.beta = beta
        current_args.gamma = gamma
        run_processor(current_args)
        print()
        print()


def main(args):
    batch_run(args)


if __name__ == "__main__":
    main(parse_args())
