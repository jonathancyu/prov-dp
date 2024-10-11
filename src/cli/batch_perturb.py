from dataclasses import dataclass

from .perturb import run_processor, parse_args
from guppy import hpy


@dataclass
class Config:
    epsilon: float
    delta: float
    alpha: float
    beta: float
    gamma: float
    eta: float
    k: int = 250


from dataclasses import asdict
from argparse import Namespace


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
    k_configurations = [
        # Varying k (probably need to add it above XD)
        Config(epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25, k=5),
        Config(epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25, k=250),
        Config(epsilon=1, delta=0.5, alpha=0.25, beta=0.25, gamma=0.25, eta=0.25, k=500),
    ]

    # Update the arguments for each configuration and run
    h = hpy()
    for config in configurations:
        print("#" * 100)
        args_dict = vars(args)
        args_dict.update(asdict(config))
        current_args = Namespace(**args_dict)
        run_processor(current_args)
        print(h.heap())
        print("\n")


def main(args):
    batch_run(args)


if __name__ == "__main__":
    main(parse_args())
