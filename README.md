# sim-fold-ipc

A GPU-ready **Incremental Potential Contact (IPC)** simulator for **endovascular
guidewire navigation**, built on [NVIDIA Warp](https://github.com/NVIDIA/warp).
An elastic rod (guidewire) is threaded through static vessel meshes while a
smooth log-barrier + continuous collision detection guarantee it never crosses a
vessel wall.

![Guidewire navigating the S-vessel](images/nav_s.png)

## About

The rod is driven by an implicit, overdamped **incremental potential**: each step
minimises `inertia + bending + stretch + contact-barrier` energy. Contact uses the
C² IPC log-barrier over the gap `distance − rod_radius`, evaluated against a Warp
`wp.Mesh` BVH, with a conservative CCD cap for a penetration-free trajectory.

Two vessel meshes are provided at neural-vessel scale (3.5 mm inner diameter): a
straight tube and an S-vessel with two opposite 60° bends. The guidewire is fed in
along a straight **rail** at constant insertion speed (or tip-guided through the
bends).

- **Week 1** — free-space kinematics: a pre-bent rod relaxes to its straight rest shape.
- **Week 2** — navigation: the guidewire threads both vessels with **zero wall penetration**.

Two rod models live in the package: a full quaternion-**Cosserat** model
(`elasticity.py`, autodiff-gradient-verified) and a well-conditioned
positions-primary **Discrete-Elastic-Rods** model (`elastic_pos.py`, used by the
working solver). See [`reports/report.md`](reports/report.md) for the full write-up,
equations, and the C-IPC scope.

> Runs on CPU (no GPU required) and uses CUDA automatically when a driver is present.

## Setup

Dependencies are managed with the [uv](https://docs.astral.sh/uv/) package manager.

```bash
# 1. install uv (if needed)
#    Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex
#    macOS / Linux:       curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. clone and sync the environment (creates .venv, installs deps)
git clone git@github.com:patoalejor/sim-fold-ipc.git
cd sim-fold-ipc
uv sync
```

`uv sync` installs `warp-lang`, `numpy`, `scipy`, `matplotlib`, `pillow`, and
`imageio` from the locked versions in `uv.lock`.

## Run the demos

Run everything with `uv run` (no manual venv activation needed):

```bash
# render the two vessel meshes -> images/  (PNG + orbit GIF)
uv run python scripts/render_meshes.py

# Week 1: pre-bent rod relaxes to straight -> images/week1_relax.*
uv run python scripts/render_week1.py

# Week 2: guidewire navigation (both vessels) -> images/nav_*.{png,gif}
uv run python scripts/demo_navigation.py          # both
uv run python scripts/demo_navigation.py straight # straight only
uv run python scripts/demo_navigation.py s        # S-vessel only
```

Outputs (static PNGs, GIF movies, and wall-clearance plots) are written to
`images/`. Generated meshes are exported to `assets/` as OBJ.

## Tests

```bash
uv run python tests/test_meshes.py          # mesh geometry checks
uv run python tests/test_rod_freespace.py   # Week 1: relax-to-straight
uv run python tests/test_navigation.py      # Week 2: penetration-free navigation
```

## Layout

```
sim_fold_ipc/   meshes, rod, elasticity (Cosserat + DER), barrier, solvers, actuation, render
scripts/        render_meshes.py, render_week1.py, demo_navigation.py
tests/          test_meshes, test_rod_freespace (W1), test_navigation (W2)
images/         renders (PNG + GIF)      reports/  milestone report (report.md)
assets/         exported OBJ meshes
```
