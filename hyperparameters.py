import numpy as np
import copy

from inverse_kinematics import run_experiment


def hyperparameters(args):
    if args.logspace is not None:
        clamps = np.logspace(start=args.clamp[0], stop=args.clamp[1], num=args.logspace)
        damps = np.logspace(start=args.damp[0], stop=args.damp[1], num=args.logspace)
        clips = np.logspace(start=args.clip_angle[0], stop=args.clip_angle[1], num=args.logspace)
    else:
        clamps = np.linspace(start=args.clamp[0], stop=args.clamp[1], num=args.linspace)
        damps = np.linspace(start=args.damp[0], stop=args.damp[1], num=args.linspace)
        clips = np.linspace(start=args.clip_angle[0], stop=args.clip_angle[1], num=args.linspace)
    temp_args = copy.deepcopy(args)
    min_arg = None
    min_dist = np.inf

    for clamp in clamps:
        for clip in clips:
            temp_args.clamp = clamp
            temp_args.clip_angle = clip
            if args.DLS:
                for damp in damps:
                    temp_args.damp = damp
                    _, _, dist = run_experiment(temp_args)
                    if dist < min_dist:
                        min_arg = (clamp, clip, damp)
                        min_dist = np.mean(dist)
            else:
                _, _, dist = run_experiment(temp_args)
            if dist < min_dist:
                min_arg = (clamp, clip)
                min_dist = np.mean(dist)
    print(f"Minimum Distance: {min_dist}, with params: {min_arg} clamp, clip, (damp)")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--period", type=int, default=5000)
    parser.add_argument("--clamp", type=float, default=[None, None], nargs=2)
    parser.add_argument("--damp", type=float, default=[1, 1], nargs=2)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--pseudo_inverse", action="store_true")
    parser.add_argument("--DLS", action="store_true")
    parser.add_argument("--max_timestep", type=int, default=30000)
    parser.add_argument("--name", type=str, default="out")
    parser.add_argument("--clip_angle", type=float, default=[1, 1], nargs=2)
    parser.add_argument("--logspace", type=int, default=None)
    parser.add_argument("--linspace", type=int, default=1)
    parser.add_argument("--PD", action="store_true")
    parser.add_argument("--P", type=float, default=[1, 1], nargs=2)
    parser.add_argument("--D", type=float, default=[1, 1], nargs=2)

    hyperparameters(parser.parse_args())


if __name__ == "__main__":
    main()
