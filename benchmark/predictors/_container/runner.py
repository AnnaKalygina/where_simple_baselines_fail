"""Launch a per-model training/inference container (Apptainer/Singularity).

Host-side dispatcher: writes the run config, bind-mounts the data directory
(read-only, holding the combined ``data_path`` h5ad) + the run's output directory
(read-write), and invokes the model's ``.sif`` with the uniform contract
``run_model.py {train|predict} /config.json`` (see ``docker/CONTRACT.md``).

The ``.sif`` is **environment-only**: our thin model code (``run_model.py`` +
``<model>_wrapper.py``, which now inlines its tiny data-load shim) is NOT baked in
but bind-mounted read-only from the host repo via ``code_binds`` and run via
``exec`` with ``entry`` (``PYTHONPATH`` set to the bind targets). Editing the
wrapper needs no rebuild. See ``docker/gears/gears.def`` and ``BUILD_ENV.md``.

We shell out to ``apptainer`` (falling back to ``singularity``) — we do NOT use
the docker Python SDK (LeoMed runs SIFs, and nothing here ever ``import docker``).

The caller passes HOST paths in the config; ``run_container`` rewrites the
input-h5ad field (``data_path``) and the ``output_path`` to the canonical
in-container mount points so the container only ever sees ``/data`` and
``/model_output``. In-container WRITE paths (``output_dir``, ``checkpoint_dir``,
``model_path``) are the constant ``/model_output`` and are passed through unchanged.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

# Canonical in-container mount points (must match docker/CONTRACT.md).
C_DATA = "/data"
C_CONFIG = "/config.json"
C_OUTPUT = "/model_output"

# Input-h5ad config field the runner rewrites to `/data/<basename>`. The canonical
# combined form uses a single `data_path` (one h5ad the wrapper slices itself).
_PATH_FIELDS = ("data_path",)


def _runtime() -> str:
    for exe in ("apptainer", "singularity"):
        if shutil.which(exe):
            return exe
    raise RuntimeError("neither 'apptainer' nor 'singularity' found on PATH")


def build_command(sif, mode: str, config_path, data_dir, output_dir, *,
                  nv: bool = True, runtime: str | None = None,
                  code_binds: dict | None = None,
                  entry: list | None = None) -> list[str]:
    """The apptainer/singularity argv that runs ``sif`` in ``mode``.

    ``code_binds`` ({host_dir: container_dest}) bind-mounts our host code read-only
    into the (env-only) container, and ``entry`` (an argv list, e.g.
    ``["python", "/app_code/gears/run_model.py"]``) is invoked via ``exec`` with
    ``PYTHONPATH`` set to the bind destinations — i.e. the sif supplies only the
    environment and our code runs live from the host (no bake-in). When ``entry``
    is omitted the sif's own ``%runscript`` is used (``run <sif> {mode} /config.json``).
    """
    rt = runtime or _runtime()
    code_binds = code_binds or {}
    cmd = [rt, "exec" if entry else "run"]
    if nv:
        cmd.append("--nv")  # expose GPUs
    cmd += [
        "--bind", f"{Path(data_dir).resolve()}:{C_DATA}:ro",
        "--bind", f"{Path(config_path).resolve()}:{C_CONFIG}:ro",
        "--bind", f"{Path(output_dir).resolve()}:{C_OUTPUT}",
    ]
    for host, dest in code_binds.items():
        cmd += ["--bind", f"{Path(host).resolve()}:{dest}:ro"]
    if entry:
        pypath = ":".join(code_binds.values())
        if pypath:
            cmd += ["--env", f"PYTHONPATH={pypath}"]
        cmd += ["--env", "PYTHONNOUSERSITE=1"]  # don't leak host ~/.local site-packages
    cmd.append(str(sif))
    cmd += (list(entry) if entry else []) + [mode, C_CONFIG]
    return cmd


def run_container(sif, mode: str, config: dict, data_dir, output_dir, *,
                  nv: bool = True, dry_run: bool = False,
                  runtime: str | None = None, timeout: int | None = None,
                  code_binds: dict | None = None, entry: list | None = None):
    """Write ``config`` to ``<output_dir>/config.json`` and run the ``.sif``.

    ``config`` uses HOST paths; the h5ad + output fields are rewritten to the
    canonical in-container mounts (the h5ad must physically live in ``data_dir``,
    the output lands in ``output_dir``). ``code_binds``/``entry`` bind-mount our
    host code into the env-only container and run it via ``exec`` (see
    ``build_command``). Returns ``(cmd, CompletedProcess|None)``; ``dry_run=True``
    returns the command without executing (used in tests).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(config)
    cfg.setdefault("mode", mode)
    for k in _PATH_FIELDS:
        if k in cfg and cfg[k]:
            cfg[k] = f"{C_DATA}/{Path(cfg[k]).name}"
    if cfg.get("output_path"):
        cfg["output_path"] = f"{C_OUTPUT}/{Path(cfg['output_path']).name}"

    # Per-mode filename: both modes used to write `config.json` into the same run
    # dir, so a predict run silently overwrote the record of how training was
    # configured. The in-container path is unchanged (bound to /config.json), so
    # this is host-side only and no wrapper needs to change.
    config_path = output_dir / f"{mode}_config.json"
    config_path.write_text(json.dumps(cfg, indent=2))

    cmd = build_command(sif, mode, config_path, data_dir, output_dir,
                        nv=nv, runtime=runtime, code_binds=code_binds, entry=entry)
    if dry_run:
        return cmd, None
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-4000:]
        raise RuntimeError(f"container {mode} failed (rc={proc.returncode}):\n{tail}")
    return cmd, proc


def _demo() -> None:
    """Dry-run smoke test: show the command for a combined-form train + predict.

    The env-only sif bind-mounts only ``docker/gears`` (the wrapper inlines its own
    data-load shim, so there is no separate harness bind).
    """
    code_binds = {"/host/repo/docker/gears": "/app_code/gears"}
    entry = ["python", "/app_code/gears/run_model.py"]
    cmd_t, _ = run_container(
        "docker/gears/gears.sif", "train",
        {"model": "gears", "dataset": "adamson16", "scenario": "UnseenPert", "fold": 0,
         "seed": 42, "regime": "UnseenPert",
         "data_path": "/host/data/adamson16/adamson16_processed.h5ad",
         "split_name": "split_UnseenPert_fold_0", "covariate_key": "cell_type",
         "train_conditions": ["AARS"], "val_conditions": ["BRCA1"],
         "test_conditions": ["TP53"],
         "output_dir": C_OUTPUT, "checkpoint_dir": C_OUTPUT},
        data_dir="/host/data/adamson16", output_dir="/tmp/_sr_demo",
        code_binds=code_binds, entry=entry, dry_run=True, runtime="apptainer")
    cmd_p, _ = run_container(
        "docker/gears/gears.sif", "predict",
        {"model_path": C_OUTPUT,
         "data_path": "/host/data/adamson16/adamson16_processed.h5ad",
         "split_name": "split_UnseenPert_fold_0", "test_conditions": ["TP53"],
         "output_path": "/tmp/_sr_demo/predictions.h5ad"},
        data_dir="/host/data/adamson16", output_dir="/tmp/_sr_demo",
        code_binds=code_binds, entry=entry, dry_run=True, runtime="apptainer")
    print("TRAIN  :", " ".join(cmd_t))
    print("PREDICT:", " ".join(cmd_p))
    print("config written to /tmp/_sr_demo/config.json")


if __name__ == "__main__":
    _demo()
