#!/usr/bin/env python3
"""Generate per-model DL vocabulary references under ``data/ref/dl_model_vocab/``.

HISTORICAL. This fed the DL fold-alignment gate (``benchmark/_fold_align.py``),
which was *gene-set aware*: when a model's per-fold test set was a subset of our
h5ad fold, the gate classified each absent perturbation as **out-of-vocabulary**
(the model structurally cannot represent that gene) vs **in-vocab-but-dropped**
(a different upstream filter). That gate and the adapter tier it checked are
deleted; this script is kept only to document how the existing artifacts under
``data/ref/dl_model_vocab/`` were produced.

Sources (confirmed in-repo / collaborator tree), one artifact per model:
  * scGPT   — GLOBAL token vocabulary `vocab.json` (~60.7k symbols), saved per run
              under `models/atheus_csb/scgpt_*/<hash>/vocab.json` (all identical).
              NOTE: uses current HGNC symbols, so it carries gene-symbol drift vs
              older dataset labels (e.g. has `QARS1`, not the legacy `QARS`).
  * GEARS   — GLOBAL gene-ontology graph genes = keys of `gene2go_all.pkl`
              (~67.8k). That file ships only in the training repo, so pass its
              path (default below); the committed JSON makes the gate self-contained.
  * PRESAGE — DATASET-specific: the measured genes (h5ad `var_names`) for each
              dataset (PRESAGE represents perturbations via gene/pathway embeddings
              over the measured genes), so one `presage_{dataset}.json` per dataset.

Schema (per file): {model, vocab_type, source, n_genes, genes: [...]}.

Read-only inputs; writes only under data/ref/dl_model_vocab/. Run in `preprocess`:
    conda run -n preprocess python data/gen_dl_vocab.py [--gears-gene2go PATH]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from benchmark import config  # noqa: E402

OUT_DIR = config.DATA_DIR / "ref" / "dl_model_vocab"
# gene2go ships with the training repo (not in this repo); override with --gears-gene2go.
DEFAULT_GENE2GO = Path(
    "/cluster/work/boeva/akalygina/literature-repos/"
    "Perturbation-Models-Outperform-Baselines/docker/gears/gene2go_all.pkl"
)
SPECIAL_TOKENS = {"<pad>", "<cls>", "<eoc>", "<mask>", "<unk>"}


def _write(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload["genes"] = sorted(payload["genes"])
    payload["n_genes"] = len(payload["genes"])
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {path.relative_to(REPO)}  (n_genes={payload['n_genes']}, source={payload['source']})")


def gen_scgpt() -> None:
    """scGPT global token vocabulary, read from any atheus_csb scGPT run (all equal)."""
    cands = sorted((config.MODELS_DIR / "atheus_csb").glob("scgpt_*/*/vocab.json"))
    if not cands:
        print("  scGPT: no vocab.json under atheus_csb/scgpt_*/ — skipping")
        return
    src = cands[0]
    vocab = json.loads(src.read_text())
    genes = [g for g in vocab if g not in SPECIAL_TOKENS and not g.startswith("<")]
    _write("scgpt.json", {
        "model": "scGPT", "vocab_type": "global_pretrained_tokens",
        "source": str(src.relative_to(config.MODELS_DIR.parent)), "genes": genes,
    })


def gen_gears(gene2go_path: Path) -> None:
    """GEARS global gene set = keys of the gene2go graph."""
    if not gene2go_path.is_file():
        print(f"  GEARS: gene2go not found at {gene2go_path} — skipping "
              f"(pass --gears-gene2go)")
        return
    with open(gene2go_path, "rb") as f:
        gene2go = pickle.load(f)
    _write("gears.json", {
        "model": "GEARS", "vocab_type": "gene_ontology_graph",
        "source": str(gene2go_path), "genes": list(gene2go.keys()),
    })


def gen_presage() -> None:
    """PRESAGE dataset-specific gene set = each dataset's measured var_names."""
    import anndata as ad
    for ds in config.DATASET_CONFIG:
        if ds == "ecoli_synthetic":
            continue
        h5 = Path(config.h5ad_path(ds))
        if not h5.exists():
            continue
        a = ad.read_h5ad(str(h5), backed="r")
        genes = [str(g) for g in a.var_names]
        try:
            a.file.close()
        except Exception:
            pass
        _write(f"presage_{ds}.json", {
            "model": "PRESAGE", "vocab_type": "dataset_measured_genes",
            "dataset": ds, "source": str(h5.relative_to(config.DATA_DIR.parent)),
            "genes": genes,
        })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gears-gene2go", type=Path, default=DEFAULT_GENE2GO)
    ap.add_argument("--only", choices=["scgpt", "gears", "presage"], default=None,
                    help="regenerate just one model's refs")
    args = ap.parse_args()
    print(f"Generating DL vocab refs -> {OUT_DIR.relative_to(REPO)}")
    if args.only in (None, "scgpt"):
        gen_scgpt()
    if args.only in (None, "gears"):
        gen_gears(args.gears_gene2go)
    if args.only in (None, "presage"):
        gen_presage()
    print("done.")


if __name__ == "__main__":
    main()
