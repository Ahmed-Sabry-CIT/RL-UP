"""Day-5 reward calibration: roll out random policies and report the scale of each
reward term so alpha/beta/gamma/delta can be chosen deliberately.

Usage: python scripts/calibrate_reward.py [--episodes 12]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from joint_fsda_env import DA, FS, JointFSDAEnv, make_demo_data  # noqa: E402

TERMS = ["d_f1", "gmean", "feat_frac", "d_syn"]


def collect(env, episodes):
    rows = []
    for ep in range(episodes):
        env.reset(seed=ep)
        while env.agents:
            a = {x: env.action_space(x).sample() for x in env.agents}
            _, _, _, _, info = env.step(a)
            rows.append([info[FS][t] for t in TERMS])
    return np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    args = ap.parse_args()
    env = JointFSDAEnv(*make_demo_data(), horizon=8, patience=99)
    R = collect(env, args.episodes)
    print(f"{'term':10s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    for j, t in enumerate(TERMS):
        print(f"{t:10s} {R[:, j].mean():8.3f} {R[:, j].std():8.3f} {R[:, j].min():8.3f} {R[:, j].max():8.3f}")
    std = R.std(axis=0) + 1e-9
    w = 1.0 / std
    w = w / w[0]                                 # alpha pinned to 1.0
    print("\nSuggested weights equalising each term's std contribution (alpha=1):")
    print(f"  alpha={w[0]:.2f} beta={w[1]:.2f} gamma={w[2]:.2f} delta={w[3]:.2f}")
    print("Note: beta*GMean is an absolute level (not a delta); keep beta modest so it does "
          "not swamp the incremental dF1 signal.")
