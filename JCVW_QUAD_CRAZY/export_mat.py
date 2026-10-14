"""Flight logs for MATLAB: <folder>/fly_*.npz -> <folder>/mat/fly_*.mat, same names.

MATLAB cannot read .npz. Each .mat holds the log's arrays under the same names,
one row per control step (t, pos, rpy, u, vbat, gyro, p_ref), and its metadata
(Ts, traj, use_phi, model). Re-run after new flights; existing files are
overwritten, and a .mat whose log was moved or deleted is removed, so <folder>/mat
always holds exactly the flights in <folder>.

    python export_mat.py              # logs/fly_*.npz      -> logs/mat/
    python export_mat.py logs/good    # logs/good/fly_*.npz -> logs/good/mat/
"""

import argparse
import glob
import os

import numpy as np
import scipy.io as sio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="logs")
    src = ap.parse_args().folder
    out_dir = os.path.join(src, "mat")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "fly_*.npz")))
    for f in files:
        d = np.load(f)
        out = {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files}
        sio.savemat(os.path.join(out_dir, os.path.basename(f)[:-4] + ".mat"), out,
                    oned_as="column")
    keep = {os.path.basename(f)[:-4] + ".mat" for f in files}
    stale = [m for m in glob.glob(os.path.join(out_dir, "fly_*.mat"))
             if os.path.basename(m) not in keep]
    for m in stale:
        os.remove(m)
    print(f"{len(files)} flight logs -> {out_dir}/"
          + (f", {len(stale)} without a log removed" if stale else ""))


if __name__ == "__main__":
    main()
