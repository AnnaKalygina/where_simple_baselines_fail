# GEARS container — build environment & route (CustomApps client)

## Design: environment-only image + bind-mounted code
`gears.sif` holds **only** the frozen environment (pytorch base + GEARS/scanpy
stack). Our code (`run_model.py`, `gears_wrapper.py` — which inlines its own tiny
data-load shim) and `gene2go_all.pkl` are **not** baked in — the host runner
bind-mounts `docker/gears/` at run time under `/app_code/gears` (see
`benchmark/predictors/_container/runner.py` and the `gears.def` `%help`).
Consequences: the `.def` needs **no `%files`/local inputs**, it builds anywhere
with internet, and editing the wrapper needs **no rebuild**.

> **Pin ≠ training recipe.** The pinned `cell-gears @e6b6d6c` supplies only the
> `GEARS_Model` (nn.Module), `PertData`, and utils; the **training loop, loss,
> `model_initialize`, checkpointing, and predict** are the vendored `GEARS` class
> inside `gears_wrapper.py`, and the **hyperparameters** are the authors'
> recommended values in `docker/gears/model.yaml`. Reproducing the training recipe
> therefore means pinning the sif *and* the wrapper + `model.yaml`, not the sif alone.

## Why not build on LeoMed nodes
LeoMed login/compute nodes run singularity-ce 3.8.3 (setuid mode) but have **no
working `--fakeroot`** for `akalygina` (`no mapping entry found in /etc/subuid`)
and no docker/podman. So a def's `%post` (apt/pip) **cannot** run there — only
convert-only builds do. This is a privilege limit, not a network one.

## Build route: the ETH CustomApps client
`customapps.leomed.ethz.ch` — a build host with internet + singularity that
supports `singularity build --fakeroot`. Anything built under
`/cluster/customapps/biomed/boeva/akalygina/` is visible **read-only on LeoMed** at
the same path (that NFS export is mounted `ro` on the cluster). SSH to it **from
your laptop** (port 22 is filtered from compute nodes). In-lab precedent: `eheiss`
builds full torch/PyG recipes there (`/cluster/customapps/biomed/boeva/eheiss/singularity/`).

**Pre-flight confirmed on the client (2026-08-13):** singularity-ce 3.8.3 ·
`/cluster/customapps/.../akalygina` writable · internet OK · `/cluster/work` **not**
mounted (irrelevant — the slim def needs only itself).

### Build steps (on the CustomApps client)
```bash
mkdir -p /cluster/customapps/biomed/boeva/akalygina/singularity/gears
cd       /cluster/customapps/biomed/boeva/akalygina/singularity/gears
# paste gears.def here (self-contained; no other files needed)
export SINGULARITY_CACHEDIR=/tmp/akalygina_cache SINGULARITY_TMPDIR=/tmp/akalygina_tmp
chmod o+x .
singularity build --fakeroot gears.sif gears.def   # ~4–5 GB, ~10–15 min
chmod o-x .
```
The sif then appears read-only on LeoMed at
`/cluster/customapps/biomed/boeva/akalygina/singularity/gears/gears.sif`.
(Watch-point: the only thing that must actually work is `--fakeroot`; eheiss builds
there, so it's expected to — if it errors with a `subuid` message, flag it.)

### Wire into the repo (on LeoMed)
`GEARSContainer.sif_path` resolves to `<repo>/docker/gears/gears.sif`. Point it at
the built sif with a **symlink** (zero-copy; `/cluster/work` is full; `*.sif` is
gitignored):
```bash
ln -s /cluster/customapps/biomed/boeva/akalygina/singularity/gears/gears.sif \
      /cluster/work/boeva/virtual_cell_reasoning/docker/gears/gears.sif
```

## Pins (resolved 2026-08-10)
| component | pin |
|---|---|
| base image | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` |
| base digest | `sha256:831247999fbf7e08f61b3e39f6d77ee434f38f6f07f769d00db451e853878067` |
| GEARS commit | `e6b6d6cb9080348fe121234082c9958cf9f17eae` (millerh1/GEARS) |
| pip deps | torch-geometric, scanpy, anndata>=0.12, pandas, numpy, scikit-learn, tqdm, omegaconf, hydra-core; exact versions → `/opt/gears_lockfile.txt` at build |

torch/CUDA are fixed by the base digest; the in-image lockfile pins the rest.

## Run (on a LeoMed GPU node, inside SLURM)
The runner invokes singularity directly (no `srun`), so submit the benchmark as a
GPU job. Quick env smoke test once the sif is in place:
```bash
srun -p gpu --gres=gpu:1 --time 00:20:00 --pty \
  singularity exec --nv docker/gears/gears.sif \
  python -c "import torch, torch_geometric, scanpy, anndata, gears; print(torch.__version__, torch.cuda.is_available())"
```
