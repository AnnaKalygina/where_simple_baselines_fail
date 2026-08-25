#!/usr/bin/env python
"""Download and preprocess the X-Atlas/Orion (Bhatt et al. 2025) dataset.

Source: HuggingFace Xaira-Therapeutics/X-Atlas-Orion — 332 parquet batches
(109 HCT116 + 223 HEK293T) and a gene_metadata.parquet vocabulary file
totaling 118 GB of compressed sparse expression data.

Scale: ~18,000 single-gene perturbations across 2 cell types (HCT116 + HEK293T)
≈ 3.2M cells. This is ~10x larger than any other dataset in the catalog, so
the pipeline is split into TWO PHASES that can be run in parallel SLURM jobs:

  Phase 1 (per cell type, parallelizable):
    python get_data.py --cell-type HCT116      ~50 min, ~80 GB peak
    python get_data.py --cell-type HEK293T     ~80 min, ~96 GB peak
  Each writes _intermediate_{CT}.h5ad: cells × shared-gene-vocab, log1p,
  raw counts in layers["counts"], per-pert downsampled. Idempotent: re-running
  with the intermediate already present is a no-op.

  Phase 2 (combine, serial):
    python get_data.py --combine               ~30 min, ~96 GB peak
  Loads both intermediates, intersects gene vocab, filters to shared perts,
  HVG selection (8192 from a 200k-cell stratified subsample), forces-include
  perturbation target genes, assigns tech_dup_split + folds, runs the
  vectorized DEG t-test (compute_degs_vectorized in _utils.py — a single
  sparse matmul per half, ~5 min total instead of ~6h via scanpy's
  per-group loop), builds pseudobulk + deg_arrays, adds bio relations,
  writes the final xatlas_orion_processed.h5ad.

  python get_data.py --all                     (runs Phase 1 sequentially then Phase 2)

Output schema matches every other dataset (DatasetStore consumes it
identically). The ONE exception: per-pert DEG dicts are written EMPTY
(would be ~140 GB of Python overhead) and the score/pval data lives in
adata.uns["scores_matrix_first_half" | "pvals_adj_matrix_first_half" |
"scores_matrix_second_half" | "pvals_adj_matrix_second_half"] as
var_names-aligned tensors. compute_deg_arrays and InterpDuplicate both
detect the matrix path automatically.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time as _time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse, vstack as sparse_vstack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _utils as u  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
# Raw parquet download cache. Overridable with XATLAS_RAW_DIR; defaults to a
# repo-relative cache so this script is portable (no hardcoded per-user path).
# The existing cluster cache lives at
# /cluster/work/boeva/akalygina/datasets/real_data/xatlas_orion — to reuse it
# without re-downloading, set XATLAS_RAW_DIR to that path (or symlink it here).
DATA_DIR = Path(os.environ.get("XATLAS_RAW_DIR", OUTPUT_DIR / "_raw_cache"))
DATASET_NAME = "xatlas_orion"
OUTPUT_PATH = OUTPUT_DIR / f"{DATASET_NAME}_processed.h5ad"

CELL_TYPES = ("HCT116", "HEK293T")
CONTROL_LABEL = "Non-Targeting"
CONTROL_LABELS_RAW = {CONTROL_LABEL, "control", "ctrl"}

# Pipeline parameters
MIN_CELLS_PRE_FILTER = 12       # drop perts with < this BEFORE downsampling
MAX_CELLS_PER_PERT_CAP = 96     # hard cap so xatlas perts don't dominate
MAX_CELLS_CONTROL = 8192
MIN_CELLS_PER_CONDITION = 4     # post-downsample minimum
N_HVG = 8192
HVG_SUBSAMPLE = 200_000
SEED = 42

# -----------------------------------------------------------------
# Memory helper — return RSS to OS aggressively. Important for the
# ~80-100 GB working set we accumulate per cell type.
# -----------------------------------------------------------------
try:
    _libc = ctypes.CDLL("libc.so.6")
    def _free_memory():
        gc.collect()
        _libc.malloc_trim(0)
except Exception:
    def _free_memory():
        gc.collect()


def _intermediate_path(cell_type: str) -> Path:
    return OUTPUT_DIR / f"_intermediate_{cell_type}.h5ad"


# ===================================================================
# Download (HuggingFace) — idempotent per-file, like the other 9 datasets.
# MUST run interactively (the cluster proxy tunnel is only up in the session);
# do NOT wire --download-only into SLURM jobs. Phase 1 assumes the cache exists.
# ===================================================================

_HF_REPO = "Xaira-Therapeutics/X-Atlas-Orion"
_HF_BASE_URL = f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main"
_HF_API_URL = f"https://huggingface.co/api/datasets/{_HF_REPO}/tree/main"
_DL_WORKERS = 2       # proxy can't handle more
_DL_MAX_RETRIES = 20  # each curl invocation only gets ~100-200 MB before the proxy cuts
_DL_RETRY_DELAY = 3


def _hf_file_list(cell_type: str | None = None) -> list[tuple[str, int]]:
    data = json.loads(urllib.request.urlopen(f"{_HF_API_URL}/data", timeout=60).read())
    files = [(f["path"], f["size"]) for f in data if f["path"].endswith(".parquet")]
    if cell_type:
        files = [(p, s) for p, s in files if cell_type in p]
    log.info("HF: %d parquet files (%.1f GB)", len(files), sum(s for _, s in files) / 1e9)
    return files


def _hf_metadata_file() -> tuple[str, int] | None:
    try:
        data = json.loads(urllib.request.urlopen(f"{_HF_API_URL}/metadata", timeout=60).read())
        for f in data:
            if f["path"].endswith("gene_metadata.parquet"):
                return f["path"], f["size"]
    except Exception as exc:
        log.warning("HF metadata query failed: %s", exc)
    return None


def _download_one(item: tuple[str, int]) -> str:
    """curl with resume (-C -); skip if already the right size; restart if oversized."""
    rel_path, expected = item
    filename = rel_path.split("/")[-1]
    url = f"{_HF_BASE_URL}/{rel_path}"
    dest = DATA_DIR / filename
    if dest.exists() and dest.stat().st_size == expected:
        return f"SKIP {filename}"
    if dest.exists() and dest.stat().st_size > expected:
        dest.unlink()
    for attempt in range(1, _DL_MAX_RETRIES + 1):
        cur = dest.stat().st_size if dest.exists() else 0
        cmd = (["curl", "-C", "-", "-L", "-s", "-f", "-o", str(dest), url]
               if 0 < cur < expected
               else ["curl", "-L", "-s", "-f", "-o", str(dest), url])
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if dest.exists():
            new = dest.stat().st_size
            if new == expected:
                return f"OK   {filename} ({new/1e6:.1f} MB)"
            if new > expected:
                dest.unlink()
            elif new <= cur and attempt % 5 == 0 and dest.exists():
                dest.unlink()  # stuck — restart
        _time.sleep(_DL_RETRY_DELAY)
    got = dest.stat().st_size if dest.exists() else 0
    return f"FAIL {filename} ({got/1e6:.1f}/{expected/1e6:.1f} MB)"


def download(cell_type: str | None = None) -> None:
    """Idempotent download of the raw parquet cache into DATA_DIR. Skips files
    already present at the expected size, so re-runs never re-download."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = _hf_file_list(cell_type=cell_type)
    if cell_type is None:
        meta = _hf_metadata_file()
        if meta:
            files.insert(0, meta)
    log.info("Downloading %d files → %s (%d workers, resume)", len(files), DATA_DIR, _DL_WORKERS)
    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=_DL_WORKERS) as pool:
        for fut in as_completed({pool.submit(_download_one, f) for f in files}):
            res = fut.result()
            ok += res.startswith("OK"); skip += res.startswith("SKIP"); fail += res.startswith("FAIL")
            if not res.startswith("SKIP"):
                log.info("  %s", res)
    log.info("Download done: OK=%d SKIP=%d FAIL=%d", ok, skip, fail)
    if fail:
        raise RuntimeError(f"{fail} download(s) failed — re-run --download-only to retry")


# ===================================================================
# Phase 1: Parquet → AnnData per cell type
# ===================================================================

def _load_gene_vocab() -> tuple[dict, list[str]]:
    """Read gene_metadata.parquet → (token_to_name dict, ordered gene_names list).

    Drops tokens whose gene_name is still an ENSG-prefixed identifier (no
    symbol assigned). Among duplicates (different tokens → same symbol), we
    keep the smallest token id.
    """
    meta_path = DATA_DIR / "gene_metadata.parquet"
    log.info("Loading gene vocabulary: %s", meta_path)
    df = pd.read_parquet(meta_path)
    df = df[~df["gene_name"].str.startswith("ENSG")].copy()
    token_to_name = dict(zip(df["gene_token_id"], df["gene_name"]))
    # Stable de-dup: keep smallest token for each gene name
    seen: set[str] = set()
    unique: dict[int, str] = {}
    for tid in sorted(token_to_name.keys()):
        name = token_to_name[tid]
        if name in seen:
            continue
        seen.add(name)
        unique[tid] = name
    gene_names = [unique[tid] for tid in sorted(unique.keys())]
    log.info("  %d unique gene symbols", len(gene_names))
    return unique, gene_names


def _build_tid_to_gidx(token_to_name: dict, gene_name_to_idx: dict) -> np.ndarray:
    """Build int32 LUT: gene_token_id → gene_index (or -1 if dropped)."""
    max_tid = max(token_to_name.keys())
    lut = np.full(max_tid + 1, -1, dtype=np.int32)
    for tid, gname in token_to_name.items():
        gidx = gene_name_to_idx.get(gname, -1)
        if gidx >= 0:
            lut[tid] = gidx
    return lut


def _condition_counts(batch_files: list[Path]) -> dict[str, int]:
    """Count cells per perturbation across all batches (cheap — reads two cols)."""
    counts: dict[str, int] = {}
    for bf in batch_files:
        df = pd.read_parquet(bf, columns=["gene_target", "pass_guide_filter"])
        df = df[df["pass_guide_filter"] == 1]
        for cond, cnt in df["gene_target"].value_counts().items():
            counts[cond] = counts.get(cond, 0) + int(cnt)
    return counts


# Per-process worker state for the parallel Pass-2 (set by the pool initializer).
_W_KEEP_PROB: dict | None = None
_W_TID2GIDX = None
_W_NGENES: int = 0
_W_GENE_NAMES = None
_W_CELLTYPE: str = ""
_W_TMPDIR = None
_BATCH_COLS = ["cell_barcode", "gene_target", "sample", "gene_token_id", "gene_expression"]


def _init_batch_worker(keep_prob, tid_to_gidx, n_genes, gene_names, cell_type, tmpdir):
    global _W_KEEP_PROB, _W_TID2GIDX, _W_NGENES, _W_GENE_NAMES, _W_CELLTYPE, _W_TMPDIR
    _W_KEEP_PROB, _W_TID2GIDX, _W_NGENES = keep_prob, tid_to_gidx, n_genes
    _W_GENE_NAMES, _W_CELLTYPE, _W_TMPDIR = gene_names, cell_type, Path(tmpdir)


def _process_batch(task: tuple):
    """Read one parquet batch → downsample → CSR → per-CELL QC/normalize/log1p →
    write a small temp h5ad. Returns the temp path (str) or None if empty.

    Runs in a worker process. ALL per-cell ops happen here (filter_cells,
    normalize_total, log1p) so the parent never runs a full-matrix pass — the
    parent only concatenates the temp h5ads on disk. Deterministic & independent
    per batch via a batch-index-keyed seed (xatlas has no upstream byte target).
    """
    import scanpy as _sc  # worker-local import is cheap (fork inherits anyway)
    bf, batch_index = task
    try:
        df = pd.read_parquet(bf, columns=_BATCH_COLS,
                             filters=[("pass_guide_filter", "==", 1)])
    except Exception:
        df = pd.read_parquet(bf, columns=_BATCH_COLS + ["pass_guide_filter"])
        df = df[df["pass_guide_filter"] == 1]
    if len(df) == 0:
        return None
    rng = np.random.RandomState(SEED + batch_index)
    probs = np.array([_W_KEEP_PROB.get(c, 0.0) for c in df["gene_target"].values])
    df = df[rng.random(len(df)) < probs].reset_index(drop=True)
    if len(df) == 0:
        return None
    token_ids_col = df["gene_token_id"].values
    expressions_col = df["gene_expression"].values
    lengths = np.fromiter((len(t) for t in token_ids_col), dtype=np.int64, count=len(df))
    all_tids = np.concatenate(token_ids_col).astype(np.int64)
    all_exprs = np.concatenate(expressions_col).astype(np.float32)
    valid = (all_tids >= 0) & (all_tids < len(_W_TID2GIDX))
    cols = np.full(len(all_tids), -1, dtype=np.int32)
    cols[valid] = _W_TID2GIDX[all_tids[valid]]
    kept = cols >= 0
    rows = np.repeat(np.arange(len(df), dtype=np.int64), lengths)[kept]
    csr = csr_matrix((all_exprs[kept], (rows, cols[kept].astype(np.int64))),
                     shape=(len(df), _W_NGENES))
    obs = pd.DataFrame({
        "cell_barcode": df["cell_barcode"].values,
        "condition": pd.Series(df["gene_target"].values).replace(
            {CONTROL_LABEL: "control", "Non-Targeting": "control"}).values,
        "sample": df["sample"].values,
    })
    obs["cell_type"] = _W_CELLTYPE.lower()
    obs["donor_id"] = DATASET_NAME
    obs.index = [f"{_W_CELLTYPE}_{batch_index:04d}_{i}" for i in range(len(df))]
    a = ad.AnnData(X=csr, obs=obs, var=pd.DataFrame(index=_W_GENE_NAMES))
    # Per-CELL QC + normalize + log1p (all row-local → identical to doing it on the
    # assembled matrix, but parallelized and never touching a full matrix).
    _sc.pp.filter_cells(a, min_genes=200)
    if a.n_obs == 0:
        return None
    _sc.pp.normalize_total(a, target_sum=1e4)
    _sc.pp.log1p(a)
    out = _W_TMPDIR / f"_p1tmp_{_W_CELLTYPE}_{batch_index:04d}.h5ad"
    u.write_h5ad_compressed(a, out)
    return str(out)


def _stream_batches_to_temps(
    cell_type: str,
    token_to_name: dict,
    gene_names: list[str],
    tmpdir: Path,
    n_workers: int = 6,
) -> list[str]:
    """Stream batch parquets for `cell_type` → per-batch processed temp h5ads.

    Workers downsample (per-condition keep_prob), build the CSR, run per-cell
    QC/normalize/log1p, and write a small temp h5ad each (see `_process_batch`).
    Returns the temp paths in batch order (deterministic row layout after
    concat). The parent never holds a full matrix.
    """
    pattern = str(DATA_DIR / f"{cell_type}_Batch*.parquet")
    batch_files = sorted(Path(p) for p in glob.glob(pattern))
    if not batch_files:
        raise FileNotFoundError(f"No batch files for {cell_type} at {pattern}")
    log.info("Loading %d batch files for %s...", len(batch_files), cell_type)

    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    tid_to_gidx = _build_tid_to_gidx(token_to_name, gene_name_to_idx)

    log.info("  pass 1/2: counting cells per condition (small-column scan)...")
    cond_counts = _condition_counts(batch_files)
    log.info("    found %d conditions (%d cells total)",
             len(cond_counts), sum(cond_counts.values()))

    valid_perts = {c: n for c, n in cond_counts.items()
                   if c not in CONTROL_LABELS_RAW and n >= MIN_CELLS_PRE_FILTER}
    mean_cells = float(np.mean(list(valid_perts.values()))) if valid_perts else 64.0
    target = min(round(mean_cells), MAX_CELLS_PER_PERT_CAP)
    log.info("    mean_cells_per_pert=%.1f → target=%d (cap=%d)",
             mean_cells, target, MAX_CELLS_PER_PERT_CAP)

    keep_prob: dict[str, float] = {}
    for c, n in cond_counts.items():
        if c in valid_perts:
            keep_prob[c] = min(1.0, target / n) if n > 0 else 1.0
        elif c in CONTROL_LABELS_RAW:
            keep_prob[c] = min(1.0, MAX_CELLS_CONTROL / n) if n > 0 else 1.0
        else:
            keep_prob[c] = 0.0  # drop perts with too few cells

    log.info("  pass 2/2: streaming %d batches with %d workers (per-batch "
             "QC/normalize/log1p → temp h5ads)...", len(batch_files), n_workers)
    t0 = _time.time()
    paths: list[tuple[int, str]] = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_batch_worker,
        initargs=(keep_prob, tid_to_gidx, n_genes, gene_names, cell_type, str(tmpdir)),
    ) as ex:
        futs = {ex.submit(_process_batch, (bf, bi)): bi
                for bi, bf in enumerate(batch_files)}
        for fut in as_completed(futs):
            p = fut.result()
            if p is not None:
                paths.append((futs[fut], p))
            done += 1
            if done % 20 == 0 or done == len(batch_files):
                log.info("    %d/%d batches done (%d non-empty, %.0fs)",
                         done, len(batch_files), len(paths), _time.time() - t0)
    # Deterministic row layout: order by batch index.
    paths.sort(key=lambda t: t[0])
    return [p for _, p in paths]


def run_phase1_one_cell_type(cell_type: str, n_workers: int = 6) -> None:
    """Phase 1 for one cell type: parquet → per-batch processed temp h5ads →
    on-disk concat into the intermediate. Memory peak ≈ one batch per worker —
    no full matrix is ever materialized, so it cannot OOM regardless of size.
    """
    from anndata.experimental import concat_on_disk
    out_path = _intermediate_path(cell_type)
    if out_path.exists():
        log.info("Phase 1 [%s]: intermediate already at %s — skipping",
                 cell_type, out_path)
        return

    np.random.seed(SEED)
    token_to_name, gene_names = _load_gene_vocab()
    tmpdir = OUTPUT_DIR / f"_p1tmp_{cell_type}"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)
    try:
        temp_paths = _stream_batches_to_temps(
            cell_type, token_to_name, gene_names, tmpdir, n_workers=n_workers)
        if not temp_paths:
            raise RuntimeError(f"Phase 1 [{cell_type}]: no non-empty batches")
        tmp_path = out_path.with_suffix(".tmp.h5ad")
        if tmp_path.exists():
            tmp_path.unlink()
        log.info("Phase 1 [%s]: concat_on_disk %d batch temps → %s",
                 cell_type, len(temp_paths), tmp_path)
        concat_on_disk(temp_paths, str(tmp_path), axis=0)
        tmp_path.rename(out_path)
        log.info("Phase 1 [%s]: done → %s", cell_type, out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===================================================================
# Phase 2: Combine intermediates + splits + DEGs + bio relations
# ===================================================================

def _shared_perts(a1: ad.AnnData, a2: ad.AnnData) -> list[str]:
    """Perturbations with >= MIN_CELLS_PER_CONDITION cells in BOTH cell types
    (excluding controls)."""
    c1 = a1.obs["condition"].value_counts()
    c2 = a2.obs["condition"].value_counts()
    shared = set(c1.index) & set(c2.index)
    valid: list[str] = []
    for cond in sorted(shared):
        if u._is_control_label(cond):
            continue
        if c1[cond] >= MIN_CELLS_PER_CONDITION and c2[cond] >= MIN_CELLS_PER_CONDITION:
            valid.append(cond)
    return valid


def _write_filtered_chunks(a, keep_mask, col_idx, tmpdir: Path, prefix: str,
                           chunk: int = 50_000) -> list[str]:
    """Stream a backed AnnData in row-chunks → per-chunk temp h5ads restricted to
    `keep_mask` rows and `col_idx` columns. Returns ordered temp paths. Peak
    memory ≈ one chunk (backed slice → in-memory block → row+col subset → write)."""
    paths: list[str] = []
    n = a.n_obs
    ci = 0
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        m = np.asarray(keep_mask[i:end], dtype=bool)
        if not m.any():
            continue
        block = a[i:end].to_memory()
        block = block[m][:, col_idx].copy()
        p = tmpdir / f"{prefix}_{ci:05d}.h5ad"
        ci += 1
        u.write_h5ad_compressed(block, p)
        paths.append(str(p))
        del block
    return paths


def run_phase2_combine() -> None:
    """Phase 2 (fully streamed): combine the two intermediates → union processed
    h5ad WITHOUT ever holding a full matrix. Backed reads + on-disk concat +
    chunked pseudobulk/DEG → peak ~tens of GB regardless of dataset size
    (cannot OOM). uns/var are attached to the on-disk file via write_elem
    (X is never re-read)."""
    from anndata.experimental import concat_on_disk, write_elem
    import h5py

    np.random.seed(SEED)
    for ct in CELL_TYPES:
        if not _intermediate_path(ct).exists():
            raise FileNotFoundError(
                f"Phase 2 requires intermediate for {ct} at "
                f"{_intermediate_path(ct)}. Run --cell-type {ct} first."
            )
    CHUNK = 50_000          # rows per chunk when building the final X on disk
    PB_CHUNK = 100_000      # rows per chunk for streamed pseudobulk/DEG

    log.info("Phase 2: opening intermediates (backed)")
    A = {ct: ad.read_h5ad(str(_intermediate_path(ct)), backed="r") for ct in CELL_TYPES}
    c0, c1 = CELL_TYPES

    # 1. Shared genes + shared perturbations (metadata only; obs is in memory).
    shared_genes = sorted(set(A[c0].var_names) & set(A[c1].var_names))
    valid_perts = _shared_perts(A[c0], A[c1])
    keep_conds = set(valid_perts) | {"control"}
    log.info("  %d shared genes; %d shared perts (>=%d cells in both)",
             len(shared_genes), len(valid_perts), MIN_CELLS_PER_CONDITION)

    # 2. HVG on a stratified backed subsample, then the HVG ∪ target union panel.
    rng = np.random.RandomState(SEED)
    sub_parts = []
    for ct in CELL_TYPES:
        a = A[ct]
        k = min(HVG_SUBSAMPLE // len(CELL_TYPES), a.n_obs)
        idx = (np.sort(rng.choice(a.n_obs, size=k, replace=False))
               if a.n_obs > k else np.arange(a.n_obs))
        s = a[idx].to_memory()
        sub_parts.append(s[:, s.var_names.get_indexer(shared_genes)].copy())
        del s
    sub = ad.concat(sub_parts)
    sc.pp.highly_variable_genes(sub, n_top_genes=N_HVG, subset=False)
    hv_mask = sub.var["highly_variable"].values.copy()
    del sub, sub_parts; _free_memory()

    # Reuse the canonical annotate + union helpers on a tiny metadata AnnData.
    meta = ad.AnnData(
        X=csr_matrix((max(len(valid_perts), 1), len(shared_genes)), dtype=np.float32),
        obs=pd.DataFrame({"condition": valid_perts or ["control"]}),
        var=pd.DataFrame(index=shared_genes),
    )
    meta.var["highly_variable"] = hv_mask
    u.annotate_perturbation_targets(meta)
    union_mask = u.union_panel_mask(meta)
    union_genes = [shared_genes[i] for i in np.where(union_mask)[0]]
    hv_union = meta.var["highly_variable"].values[union_mask]
    target_union = meta.var["is_perturbation_target"].values[union_mask]
    log.info("  union panel: %d genes (HVG=%d, target=%d)",
             len(union_genes), int(hv_union.sum()), int(target_union.sum()))

    tmpdir = OUTPUT_DIR / "_p2tmp"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)
    final_tmp = OUTPUT_PATH.with_suffix(".tmp.h5ad")
    try:
        # 3. Build the final union-panel X on disk: per-CT chunked row+col subset
        #    → concat_on_disk. Never materializes a full matrix.
        chunk_paths: list[str] = []
        for ct in CELL_TYPES:
            a = A[ct]
            keep = a.obs["condition"].isin(keep_conds).values
            col_idx = a.var_names.get_indexer(union_genes)
            log.info("  %s: chunking %d/%d kept cells → union cols", ct, int(keep.sum()), a.n_obs)
            chunk_paths += _write_filtered_chunks(a, keep, col_idx, tmpdir, ct, CHUNK)
        for ct in CELL_TYPES:
            A[ct].file.close()
        if final_tmp.exists():
            final_tmp.unlink()
        log.info("  concat_on_disk %d chunks → %s", len(chunk_paths), final_tmp)
        concat_on_disk(chunk_paths, str(final_tmp), axis=0)
        shutil.rmtree(tmpdir, ignore_errors=True)

        # 4. Open the final file backed; attach var flags; assign splits (obs-only).
        adata = ad.read_h5ad(str(final_tmp), backed="r")
        adata.uns["dataset_name"] = DATASET_NAME
        adata.var["highly_variable"] = hv_union
        adata.var["is_perturbation_target"] = target_union
        log.info("  combined: %d cells x %d genes", adata.n_obs, len(union_genes))
        u.assign_tech_dup_split(adata, bin_col="cell_type", seed=SEED)
        # NOTE (deferred convention fix): the other multi-bin datasets use the
        # perturbench seed-0 block partition for UnseenPert
        # (assign_split_folds_unseen_pert_multibin), tying UnseenPert to UnseenBoth.
        # xatlas keeps the single-bin round-robin (seed=SEED) below because it has
        # NO DL models to align to, and switching would regenerate the h5ad and
        # invalidate ~55 GB of saved baseline predictions/results. The verifier's
        # L1d convention check therefore reports xatlas as INFO, not a failure. To
        # adopt the convention later: swap to assign_split_folds_unseen_pert_multibin
        # (seed=0), add the min_cells=3 gene QC, and re-score.
        u.assign_split_folds_unseen_pert(adata, n_folds=5, seed=SEED)
        u.assign_split_folds_unseen_both(adata, bin_col="cell_type", n_folds=5, seed=0)
        u.assert_split_invariants(adata)

        # 5. Streamed pseudobulk + DEGs over the backed X (chunked).
        adata.uns["pseudobulk"] = u.compute_pseudobulk(
            adata, bin_col="cell_type", chunk_size=PB_CHUNK)
        u.compute_degs_vectorized(adata, half="first_half", bin_col="cell_type",
                                  dataset_name=DATASET_NAME, chunk_size=PB_CHUNK)
        u.compute_degs_vectorized(adata, half="second_half", bin_col="cell_type",
                                  dataset_name=DATASET_NAME, chunk_size=PB_CHUNK)
        adata.uns["deg_arrays"] = u.compute_deg_arrays(
            adata, bin_col="cell_type", dataset_name=DATASET_NAME)

        # 6. Attach obs/var/uns to the on-disk file (X is NOT rewritten).
        #    Snapshot the in-memory pieces, then CLOSE the backed handle so h5py
        #    can reopen the file r+ (X is no longer needed — pseudobulk/DEG done).
        obs_df, var_df = adata.obs.copy(), adata.var.copy()
        uns_d = dict(adata.uns)
        n_final = adata.n_obs
        adata.file.close()
        del adata; _free_memory()
        log.info("  attaching obs/var/uns via write_elem (no X rewrite)")
        with h5py.File(str(final_tmp), "r+") as f:
            for key, val in (("obs", obs_df), ("var", var_df),
                             ("uns", uns_d)):
                if key in f:
                    del f[key]
                write_elem(f, key, val)
        if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
            OUTPUT_PATH.unlink()
        final_tmp.rename(OUTPUT_PATH)
        log.info("Phase 2: DONE → %s (%d cells x %d genes)",
                 OUTPUT_PATH, n_final, len(union_genes))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if final_tmp.exists():
            final_tmp.unlink()

    # Delete the consumed intermediates (disk is the binding constraint).
    for ct in CELL_TYPES:
        ip = _intermediate_path(ct)
        if ip.exists():
            log.info("Phase 2: deleting consumed intermediate %s", ip)
            ip.unlink()


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="xatlas_orion preprocessing")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--cell-type", choices=list(CELL_TYPES),
        help="Phase 1: build intermediate for ONE cell type (parallelizable)",
    )
    grp.add_argument(
        "--combine", action="store_true",
        help="Phase 2: combine both intermediates → final processed h5ad",
    )
    grp.add_argument(
        "--all", action="store_true",
        help="Run Phase 1 for both cell types (sequentially) then Phase 2",
    )
    grp.add_argument(
        "--download-only", action="store_true",
        help="Only download the raw parquet cache (run INTERACTIVELY — needs the "
             "proxy tunnel; not for SLURM). Idempotent; skips files already present.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=6,
        help="Parallel worker processes for Phase-1 parquet streaming (default 6)",
    )
    args = parser.parse_args()

    if args.download_only:
        download()
    elif args.cell_type:
        run_phase1_one_cell_type(args.cell_type, n_workers=args.n_workers)
    elif args.combine:
        run_phase2_combine()
    else:  # --all
        for ct in CELL_TYPES:
            run_phase1_one_cell_type(ct, n_workers=args.n_workers)
        run_phase2_combine()


if __name__ == "__main__":
    main()
