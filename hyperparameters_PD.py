import numpy as np
import copy

from inverse_kinematics import run_experiment


def hyperparameters(args):
    if args.logspace is not None:
        P = np.logspace(start=args.P[0], stop=args.P[1], num=args.logspace)
        D = np.logspace(start=args.D[0], stop=args.P[1], num=args.logspace)
    else:
        P = np.linspace(start=args.P[0], stop=args.P[1], num=args.linspace)
        D = np.linspace(start=args.D[0], stop=args.P[1], num=args.linspace)
    temp_args = copy.deepcopy(args)
    min_arg = None
    min_dist = np.inf

    for dec in range(1, 11):
        for p in P:
            for d in D:
                temp_args.P = p
                temp_args.D = d
                temp_args.decimation = dec
                _, _, dist = run_experiment(temp_args)
                if dist < min_dist:
                    min_arg = (p, d)
                    min_dist = np.mean(dist)
        print(f"Minimum Distance: {min_dist}, with params: {min_arg} P, D")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--period", type=int, default=5000)
    parser.add_argument("--clamp", type=float, default=None)
    parser.add_argument("--damp", type=float, default=1)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--pseudo_inverse", action="store_true")
    parser.add_argument("--DLS", action="store_true")
    parser.add_argument("--max_timestep", type=int, default=30000)
    parser.add_argument("--name", type=str, default="out")
    parser.add_argument("--clip_angle", type=float, default=1)
    parser.add_argument("--logspace", type=int, default=None)
    parser.add_argument("--linspace", type=int, default=1)
    parser.add_argument("--PD", action="store_true")
    parser.add_argument("--P", type=float, default=[1, 1], nargs=2)
    parser.add_argument("--D", type=float, default=[1, 1], nargs=2)
    parser.add_argument("--decimation", type=int, default=5)

    hyperparameters(parser.parse_args())


if __name__ == "__main__":
    main()
