import contextlib
from copy import deepcopy

from src.cli.utility import run_processor, parse_args


def batch_run(args):
    args.delta = 1.0  # Allocate all privacy budget to pruning
    for epsilon_1 in [15, 20, 25, 30, 35, 40, 45]:
        for alpha in [0.1, 0.5, 0.9]:
            for beta in [0.1, 0.5, 0.9]:
                for gamma in [0.1, 0.5, 0.9]:
                    current_args = deepcopy(args)
                    print(f'(0) beginning epsilon_1={epsilon_1}, alpha={alpha}, beta={beta}, gamma={gamma}')
                    current_args.epsilon = epsilon_1
                    current_args.alpha = alpha
                    current_args.beta = beta
                    current_args.gamma = gamma
                    run_processor(current_args)
                    print()
                    print()


def main(args):
    with open("../../output/output.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            batch_run(args)


if __name__ == '__main__':
    main(parse_args())
