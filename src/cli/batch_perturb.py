import itertools
from copy import deepcopy

from .perturb import run_processor, parse_args


def batch_run(args):
    epsilon_1s = [0.1, 0.5, 10, 15]
    epsilon_2s = [0.1, 0.5, 10, 15]
    alphas = [0, 0.5]
    betas = [0, 0.5]
    gammas = [0, 0.5]

    configurations = itertools.product(epsilon_1s, epsilon_2s, alphas, betas, gammas)

    for epsilon_1, epsilon_2, alpha, beta, gamma in configurations:
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
