import mujoco
import mujoco.viewer

import numpy as np
from PIL import Image
from time import time


m = mujoco.MjModel.from_xml_path("double_y.xml")
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, height=320, width=400)

paused = False

num_ends = 4
num_joints = 16


def key_callback(keycode):
    if chr(keycode) == ' ':
        global paused
        paused = not paused


def clamp_mag(dist, d_max):
    clamped = np.zeros_like(dist)
    if d_max is None:
        return dist
    for i, e in enumerate(dist):
        if np.linalg.norm(e) < d_max:
            clamped[i] = e
        else:
            clamped[i] = (e / np.linalg.norm(e)) * d_max
    return clamped


def run(args):
    frame_list = []
    times = []
    step = 0
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running() and step < args.max_timestep:
            if not paused:
                mujoco.mj_step(m, d)
                viewer.sync()
                if step % 200 == 0:
                    renderer.update_scene(d)
                    frame_list.append(Image.fromarray(renderer.render()))
                target_pos = np.asarray([d.geom(f"target{i}").xpos for i in range(num_ends)])
                end_pos = np.asarray([d.site(f"end{i}").xpos for i in range(num_ends)])
                e_dist = target_pos - end_pos

                J = np.zeros((3 * num_ends, num_joints))
                jacp = np.zeros((3, m.nv))
                jacr = np.zeros((3, m.nv))
                J_alt = np.zeros_like(J)
                J_alt2 = np.zeros_like(J_alt)
                for i in range(num_ends):
                    id = m.joint(f"jnt_end{i}").bodyid
                    mujoco.mj_jac(m=m, d=d, jacp=jacp, jacr=jacr, point=end_pos[i][:, np.newaxis], body=id)
                    J_alt[i * 3:(i + 1) * 3] += jacr[:, :num_joints]
                    J_alt2[i * 3:(i + 1) * 3] += jacp[:, :num_joints]
                J = J_alt2

                current_time = time()
                e_clamped = clamp_mag(e_dist, args.clamp).flatten()
                if args.transpose:
                    J_T = J.T
                    temp = (J @ J_T @ e_clamped)
                    alpha = (e_clamped @ temp) / (temp @ temp)
                    delta_theta = alpha * J_T @ e_clamped
                elif args.pseudo_inverse:
                    inv = np.linalg.pinv(J)
                    delta_theta = inv @ e_clamped
                elif args.DLS:
                    J_T = J.T
                    delta_theta = J_T @ (J @ J_T + args.damp ** 2 * np.eye(3 * num_ends)) @ e_clamped
                else:
                    delta_theta = [0 for _ in range(num_joints)]

                times.append(time() - current_time)

                delta_theta = np.clip(delta_theta, -args.clip_angle, args.clip_angle)
                for i in range(num_joints):
                    d.actuator(d.joint(i).name+"act").ctrl = delta_theta[i]

                for i in range(num_ends):
                    if step % args.period < args.period // 2:
                        d.actuator(f"t{i}v").ctrl = 0.05
                    else:
                        d.actuator(f"t{i}v").ctrl = -0.05

                step += 1
    return frame_list, times


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=5000)
    parser.add_argument("--clamp", type=float, default=None)
    parser.add_argument("--damp", type=float, default=1)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--pseudo_inverse", action="store_true")
    parser.add_argument("--DLS", action="store_true")
    parser.add_argument("--max_timestep", type=int, default=50000)
    parser.add_argument("--name", type=str, default="out")
    parser.add_argument("--clip_angle", type=float, default=1)

    args = parser.parse_args()
    current_time = time()
    frame_list, compute_times = run(args)
    compute_time = np.mean(compute_times)
    print(f"Total Time: {time() - current_time}. "
          f"\nApproximately {compute_time * 1e6} microseconds to compute delta_theta")
    frame_list[0].save(f"{args.name}.gif", save_all=True, append_images=frame_list[1:])


if __name__ == "__main__":
    main()
