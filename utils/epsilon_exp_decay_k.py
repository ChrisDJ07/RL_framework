"""
Compute exponential epsilon decay rate (k) for schedules of the form:

    epsilon(t) = eps_min + (eps_start - eps_min) * exp(-k * t)

Notes:
- With this equation, epsilon approaches eps_min asymptotically.
- It cannot equal eps_min exactly at finite t.
- This tool therefore solves for k using a target epsilon at the final episode.
"""

import argparse
import math


def epsilon_at(step, eps_start, eps_min, k):
    return eps_min + (eps_start - eps_min) * math.exp(-k * step)


def solve_k(eps_start, eps_min, steps, eps_target):
    if steps <= 0:
        raise ValueError("episodes/steps must be > 0.")
    if not (eps_start > eps_min):
        raise ValueError("epsilon_start (or epsilon_max) must be > epsilon_min.")
    if not (eps_min < eps_target < eps_start):
        raise ValueError("target final epsilon must satisfy: epsilon_min < target < epsilon_start.")

    ratio = (eps_target - eps_min) / (eps_start - eps_min)
    return -math.log(ratio) / steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute exponential epsilon decay rate k for RL training."
    )
    parser.add_argument(
        "--epsilon-start",
        "--epsilon-max",
        dest="epsilon_start",
        type=float,
        required=True,
        help="Initial epsilon (alias: --epsilon-max).",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        required=True,
        help="Minimum epsilon.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="Number of episodes/steps over which decay is applied.",
    )
    parser.add_argument(
        "--target-final-epsilon",
        type=float,
        default=None,
        help=(
            "Desired epsilon at the final episode. If omitted, "
            "the script uses epsilon_min + final_gap."
        ),
    )
    parser.add_argument(
        "--final-gap",
        type=float,
        default=None,
        help=(
            "Absolute gap from epsilon_min at final episode "
            "(eps_target = epsilon_min + final_gap)."
        ),
    )
    parser.add_argument(
        "--final-gap-ratio",
        type=float,
        default=0.01,
        help=(
            "If --target-final-epsilon and --final-gap are omitted, "
            "use final_gap_ratio * (epsilon_start - epsilon_min). Default: 0.01."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    eps_start = float(args.epsilon_start)
    eps_min = float(args.epsilon_min)
    episodes = int(args.episodes)

    if args.target_final_epsilon is not None:
        eps_target = float(args.target_final_epsilon)
    else:
        if args.final_gap is not None:
            final_gap = float(args.final_gap)
        else:
            final_gap = float(args.final_gap_ratio) * (eps_start - eps_min)
        eps_target = eps_min + final_gap

    k = solve_k(eps_start, eps_min, episodes, eps_target)

    eps_final = epsilon_at(episodes, eps_start, eps_min, k)
    eps_half = epsilon_at(max(1, episodes // 2), eps_start, eps_min, k)

    print("Exponential Epsilon Decay Solver")
    print(f"epsilon_start      : {eps_start:.8f}")
    print(f"epsilon_min        : {eps_min:.8f}")
    print(f"episodes           : {episodes}")
    print(f"target_final_eps   : {eps_target:.8f}")
    print(f"k (exp decay rate) : {k:.10f}")
    print(f"epsilon@half       : {eps_half:.8f}")
    print(f"epsilon@final      : {eps_final:.8f}")
    print("")
    print("Use this in config:")
    print(f"  \"epsilon_schedule\": \"exp\"")
    print(f"  \"epsilon_exp_decay_rate\": {k:.10f}")


if __name__ == "__main__":
    main()

# python utils/epsilon_exp_decay_k.py --epsilon-start 4.0 --epsilon-min 0.05 --episodes 10000 --target-final-epsilon 0.051