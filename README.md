# KoopmanDPC

Koopman-based data-driven predictive control for a quadrotor. MATLAB scripts for
simulation/plant setup, plus a Python model-learning script (PyTorch).

## Environment (WSL)

The Python code runs in a virtual environment stored in the WSL home directory
(`~/jcvw-venv`), **not** inside this project folder. Keeping it out of `/mnt/c`
avoids path/symlink issues and is faster.

### Activate the venv

```bash
source ~/jcvw-venv/bin/activate
```

Your prompt will show `(jcvw-venv)` when it's active. To leave it:

```bash
deactivate
```

### Run the model-learning script

```bash
cd /mnt/c/TUe/Job/JCVW_QUAD
source ~/jcvw-venv/bin/activate
python model_learning_quad.py
```

### In VS Code (WSL-connected)

Ctrl+Shift+P -> "Python: Select Interpreter" -> enter `~/jcvw-venv/bin/python`.
After that, the Run button uses the right environment automatically.

## Recreating the venv from scratch

If the venv is ever lost or broken:

```bash
python3 -m venv ~/jcvw-venv
source ~/jcvw-venv/bin/activate
pip install numpy scipy matplotlib torch
```

The Python dependencies are: `numpy`, `scipy`, `matplotlib`, `torch`.
The commands above install the CUDA/GPU build of torch. For a smaller CPU-only
build, append `--index-url https://download.pytorch.org/whl/cpu` to the pip command.

## Data

The `data/` folder (large `.mat` files) is **git-ignored** and kept local only.
The scripts expect these files present in `data/` at runtime; they are not part of
the repository.

## Files

- `model_learning_quad.py` - Python model learning (PyTorch).
- `KDPC_quad_sim.m` - Koopman DPC quadrotor simulation.
- `quad_plant.m`, `calibrate_plant.m` - plant model and calibration.
- `Terminal_set_quad.m` - terminal set computation.
- `load_data.m` - data loading helper.
