"""Shared DL fold-alignment leaf — the single source of truth for matching
externally-trained DL predictions (scGPT/GEARS/PRESAGE) to our h5ad folds.

This module is a low-level LEAF: it imports only stdlib + numpy + `benchmark.config`
(and `anndata` lazily). Both the verifier (`benchmark.verify`, L6/L6b) and the DL
adapter (`benchmark.predictors.dl_adapter`) import from here, so the canonicalization
and fold-resolution logic can never diverge between "the gate" and "the verifier",
and there is no verify↔dl_adapter import cycle.

Contents:
  * MANIFEST access + scenario/fold resolution (moved from dl_adapter):
    load_manifest, parse_manifest_dataset, get_model_folds, resolve_canonical_folds,
    _read_test_conditions, _fold_from_metadata, normalize_combo_label.
  * Canonical pert keys + declared-test-set readers (moved from verify):
    match_key, canon_set, dl_declared_test_set.
  * Vocabulary-aware gate: load_model_vocab, fold_alignment_verdict.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from benchmark.config import DATA_DIR, DATASET_CONFIG, MODELS_DIR

log = logging.getLogger(__name__)

PREDICTIONS_DIR = MODELS_DIR / "atheus_csb"
MANIFEST_PATH = PREDICTIONS_DIR / "MANIFEST.json"

# Legacy MANIFEST scenario codes (s1..s6) → PascalCase.
LEGACY_TO_PASCAL = {
    "s1": "UnseenPert", "s2": "UnseenCell", "s3": "UnseenBoth",
    "s4": "UnseenPair", "s5": "UnseenDose", "s6": "UnseenCombo",
}

# Canonical-key helpers (guide-/dose-insensitive, drop control/junk).
_GUIDE_SUFFIX = re.compile(r"_\d+$")
_JUNK = {"", "nan", "none", "null", "*", "?"}


# ===================================================================
# Canonical perturbation keys (moved from verify.py)
# ===================================================================
def match_key(label: str) -> str:
    """Guide- and dose-insensitive gene-combo key for matching OUR (guide-free)
    ko_names against DL prediction labels.

    DL prediction files encode per-guide variants either as a separate numeric
    '+N' token ('FDPS+HUS1+2') or a '_N' suffix ('FDPS_2'); our ko_names carry no
    guides (stripped in get_data), so this matcher absorbs the DL encoding: drop
    the dose, drop pure-numeric guide tokens, strip any '_N' suffix, drop
    control/junk, sort. Both 'FDPS+HUS1+2' and our 'FDPS+HUS1' → 'FDPS+HUS1'.
    """
    out: List[str] = []
    for t in str(label).split("@")[0].replace(";", "+").split("+"):
        t = t.strip()
        if not t or t.isdigit():            # pure-numeric '+N' guide token
            continue
        t = _GUIDE_SUFFIX.sub("", t)         # '_N' guide suffix
        if t and t.lower() not in _JUNK and "control" not in t.lower():
            out.append(t)
    return "+".join(sorted(out))


def canon_set(conds) -> Set[str]:
    """Canonical pert-key set for DL-vs-h5ad fold comparison (drops control/junk → ""
    → excluded), collapsing the DL guide encoding ('+N' or '_N')."""
    return {k for k in (match_key(x) for x in conds) if k}


# ===================================================================
# MANIFEST access + scenario/fold resolution (moved from dl_adapter.py)
# ===================================================================
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"models": {}}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def parse_manifest_dataset(manifest_ds: str) -> Tuple[str, Optional[str]]:
    """Convert MANIFEST 'jiang24_s1' → ('jiang24', 'UnseenPert').

    For single-scenario datasets (no suffix), returns that dataset's only
    scenario. Returns (manifest_ds, None) if the dataset is unknown.
    """
    parts = manifest_ds.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower().startswith("s") and parts[1][1:].isdigit():
        scenario = LEGACY_TO_PASCAL.get(parts[1].lower())
        return parts[0], scenario
    if manifest_ds in DATASET_CONFIG:
        scenarios = list(DATASET_CONFIG[manifest_ds]["scenarios"].keys())
        return manifest_ds, scenarios[0] if scenarios else None
    return manifest_ds, None


def _fold_from_metadata(pred_path: Path) -> Optional[int]:
    meta = pred_path.parent / "inference_metadata.json"
    if not meta.exists():
        return None
    try:
        with open(meta) as f:
            md = json.load(f)
        split_name = md.get("prediction_config", {}).get("split_name", "")
        if "_fold_" in split_name:
            return int(split_name.split("_fold_")[1])
    except Exception as e:
        log.debug("could not parse fold from %s: %s", meta, e)
        return None
    return None


def normalize_combo_label(c: str) -> str:
    """Guide-resolved canonical key for a (combo) condition, robust to the encoding
    mismatch between the DL prediction files and our labels.

    The DL prediction h5ads write per-guide suffixes with '+' (the same char as the
    combo separator): our 'FDPS+HUS1_2' / 'FDPS_2+HUS1_2' appear as 'FDPS+HUS1+2' /
    'FDPS+2+HUS1+2'. We split on '+', reattach any pure-numeric token as a '_N' guide
    suffix of the preceding gene, then sort — collapsing BOTH encodings to one key
    while keeping guide variants distinct (FDPS+HUS1 != FDPS_2+HUS1). No-op for
    guide-free combos and single genes (gene symbols are never purely numeric), so
    all other datasets are unaffected.
    """
    genes: list = []
    for t in str(c).split("+"):
        if t == "":
            continue
        if t.isdigit() and genes:
            genes[-1] = f"{genes[-1]}_{t}"
        else:
            genes.append(t)
    return "+".join(sorted(genes))


def _read_test_conditions(pred_path: Path) -> set:
    """Set of test condition labels in a model's predictions h5ad (backed read).

    Returns an EMPTY set if the file is unreadable (missing / permission-denied /
    corrupt) so one inaccessible model — e.g. an ACL-locked external drop — does
    not break canonical fold resolution for the OTHER models. The caller drops
    empty entries before resolving."""
    import anndata as ad
    try:
        h = ad.read_h5ad(str(pred_path), backed="r")
    except Exception as e:  # noqa: BLE001 — OSError/PermissionError/h5py errors
        log.warning("_read_test_conditions: cannot read %s (%s) — excluding this "
                    "model-fold from fold resolution", pred_path, e)
        return set()
    try:
        col = "condition" if "condition" in h.obs.columns else h.obs.columns[0]
        return {str(c) for c in h.obs[col].astype(str).unique()}
    finally:
        try:
            h.file.close()
        except Exception:
            pass


# Cache: (dataset, scenario) -> resolution dict, so the 3 adapters share one read.
_FOLD_RESOLUTION_CACHE: Dict[Tuple[str, str], dict] = {}


def resolve_canonical_folds(
    dataset: str, scenario: str, manifest: Optional[dict] = None,
    reference: str = "gears",
) -> dict:
    """Resolve each model's source fold to a CANONICAL fold by EXACT containment.

    DL models share one fold partition per dataset, but a model may label the
    fold indices differently (scGPT's fold0/fold1 are swapped vs GEARS on
    replogle20 & wessels23). We anchor the canonical fold indices to a reference
    model (GEARS) and assign every other model-fold to the reference fold it
    overlaps — requiring it overlap EXACTLY ONE reference fold and be disjoint
    from the rest (raise otherwise; no fuzzy/best-overlap guessing).

    Returns dict with:
      canonical : {canonical_fold: set(test conditions)}  (union across models)
      remap     : {(model, source_fold): canonical_fold}
      test_sets : {(model, source_fold): set}
      ref, ref_folds
    Asserts the canonical folds are mutually disjoint (a clean exact partition).
    """
    key = (dataset, scenario)
    if key in _FOLD_RESOLUTION_CACHE:
        return _FOLD_RESOLUTION_CACHE[key]

    manifest = manifest or load_manifest()
    fbm = get_model_folds(manifest, dataset, scenario)
    if not fbm:
        raise FileNotFoundError(
            f"resolve_canonical_folds: no MANIFEST models for {dataset}/{scenario}"
        )
    test_sets: Dict[Tuple[str, int], set] = {}
    for model, folds in fbm.items():
        for f in folds:
            test_sets[(model, f["fold"])] = _read_test_conditions(f["predictions_path"])

    # Drop unreadable/empty model-folds (e.g. an ACL-locked PRESAGE drop) so one
    # inaccessible model does not break resolution for the readable ones.
    unreadable = [k for k, v in test_sets.items() if not v]
    if unreadable:
        log.warning("resolve_canonical_folds %s/%s: %d model-fold(s) unreadable/empty "
                    "— excluded from fold resolution: %s",
                    dataset, scenario, len(unreadable), sorted(unreadable)[:6])
        for k in unreadable:
            del test_sets[k]
    if not test_sets:
        raise FileNotFoundError(
            f"resolve_canonical_folds: no readable predictions for {dataset}/{scenario}")
    readable_models = {m for (m, _f) in test_sets}

    ref = reference if reference in readable_models else sorted(readable_models)[0]
    ref_folds = sorted(f for (m, f) in test_sets if m == ref)
    ref_sets = {rf: test_sets[(ref, rf)] for rf in ref_folds}

    # Exact-containment orientation requires the reference folds to be mutually
    # DISJOINT (a clean partition). That holds for UnseenPert/UnseenBoth/UnseenCombo
    # but NOT for UnseenPair, which holds out (bin,ko) PAIRS — so a gene recurs in
    # several folds' condition sets and no model fold maps to exactly one reference
    # fold. When the reference folds overlap, fall back to the metadata fold index
    # (which get_model_folds already trusts): model fold N -> canonical fold N.
    ref_disjoint = all(
        not (ref_sets[i] & ref_sets[j])
        for ii, i in enumerate(ref_folds) for j in ref_folds[ii + 1:]
    )
    if not ref_disjoint:
        log.warning(
            "%s/%s: reference (%s) folds are not mutually disjoint (overlapping "
            "per-fold conditions — expected for UnseenPair); mapping each model "
            "fold to its own metadata index instead of exact-containment orientation.",
            dataset, scenario, ref,
        )
        remap = {(model, sf): sf for (model, sf) in test_sets}
        canonical = {}
        for (model, sf), s in test_sets.items():
            canonical.setdefault(sf, set()).update(s)
        res = {"canonical": canonical, "remap": remap, "test_sets": test_sets,
               "ref": ref, "ref_folds": ref_folds, "fallback": True}
        _FOLD_RESOLUTION_CACHE[key] = res
        return res

    remap: Dict[Tuple[str, int], int] = {}
    for (model, sf), s in test_sets.items():
        if model == ref:
            remap[(model, sf)] = sf
            continue
        hits = [rf for rf in ref_folds if len(s & ref_sets[rf]) > 0]
        if len(hits) != 1:
            sizes = {rf: len(s & ref_sets[rf]) for rf in ref_folds}
            raise ValueError(
                f"{dataset}/{scenario}: {model} source-fold {sf} overlaps "
                f"{len(hits)} reference folds {hits} (overlaps={sizes}) — cannot "
                f"orient by exact containment. Refusing to guess."
            )
        remap[(model, sf)] = hits[0]

    canonical: Dict[int, set] = {rf: set() for rf in ref_folds}
    for (model, sf), s in test_sets.items():
        canonical[remap[(model, sf)]] |= s
    for i in ref_folds:
        for j in ref_folds:
            if i < j and (canonical[i] & canonical[j]):
                clash = sorted(canonical[i] & canonical[j])
                raise ValueError(
                    f"{dataset}/{scenario}: canonical fold{i} ∩ fold{j} is non-empty "
                    f"({len(clash)}: {clash[:5]}) — folds are NOT a clean partition."
                )

    res = {"canonical": canonical, "remap": remap, "test_sets": test_sets,
           "ref": ref, "ref_folds": ref_folds, "fallback": False}
    _FOLD_RESOLUTION_CACHE[key] = res
    return res


def get_model_folds(
    manifest: dict, dataset: str, scenario: str,
    model_filter: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """Return {model_name: [{fold, predictions_path}, ...]} for a (dataset, scenario)."""
    result: Dict[str, List[dict]] = {}
    for key, entry in manifest.get("models", {}).items():
        # Multi-scenario datasets (jiang24, mcfaline23, replogle22) encode the
        # scenario in the MANIFEST *key* suffix (e.g. "gears_jiang24_s1"), while
        # entry["dataset"] is the bare base name ("jiang24"). Prefer the key
        # suffix; fall back to parse_manifest_dataset for single-scenario datasets
        # (e.g. "gears_adamson16", no suffix).
        base_ds = entry["dataset"]
        prefix = f"{entry['model']}_{base_ds}_"
        sc = (LEGACY_TO_PASCAL.get(key[len(prefix):].lower())
              if key.startswith(prefix) else None)
        if sc is None:
            base_ds, sc = parse_manifest_dataset(entry["dataset"])
        if base_ds != dataset or sc != scenario:
            continue
        model = entry["model"]
        if model_filter and model != model_filter:
            continue
        folds: List[dict] = []
        seen: set = set()
        for fold_entry in entry.get("folds", []):
            pred_h5ad = fold_entry.get("predictions_h5ad", "")
            if not pred_h5ad:
                continue
            pred_path = PREDICTIONS_DIR / pred_h5ad
            if not pred_path.is_file():
                continue
            # Prefer the real fold from inference_metadata.json. Some manifests
            # (jiang24/mcfaline23/replogle22) carry a null fold-level split_name,
            # so fall back to the manifest split_name suffix only when metadata
            # is unavailable — and skip (don't crash) if neither yields a fold.
            real_fold = _fold_from_metadata(pred_path)
            if real_fold is not None:
                fold = real_fold
            else:
                split_name = fold_entry.get("split_name") or ""
                try:
                    fold = int(split_name.split("_")[-1])
                except (ValueError, AttributeError):
                    log.warning(
                        "get_model_folds: skip fold for model %r (%s/%s): no "
                        "metadata fold and split_name %r lacks a trailing "
                        "'_<int>'", model, dataset, scenario, split_name)
                    continue
            if fold in seen:
                continue
            seen.add(fold)
            folds.append({"fold": fold, "predictions_path": pred_path})
        if folds:
            result[model] = folds
    return result


# ===================================================================
# DL declared test set (moved from verify.py)
# ===================================================================
def dl_declared_test_set(pred_path) -> Set[str]:
    """The DECLARED (canonical) test perturbation set for one DL fold.

    Reads the declared split, NOT the predicted obs — GEARS/scGPT emit a
    perts×cells CROSS PRODUCT, so their `obs['condition']` lists every perturbation
    rather than the fold's held-out set, and comparing that to our fold test set
    would spuriously fail. Source priority:
      1. metadata.json            config.test_conditions
      2. inference_metadata.json  prediction_config/docker_config.test_conditions
      3. predicted obs            (last resort; correct for sparse PRESAGE)
    """
    import anndata as ad
    parent = Path(pred_path).parent
    for fname, getter in (
        ("metadata.json", lambda j: (j.get("config", {}) or {}).get("test_conditions")),
        ("inference_metadata.json",
         lambda j: ((j.get("prediction_config", {}) or j.get("docker_config", {}) or {})
                    .get("test_conditions"))),
    ):
        fp = parent / fname
        if fp.exists():
            try:
                tc = getter(json.loads(fp.read_text())) or []
            except Exception:
                tc = []
            if tc:
                return canon_set(tc)
    try:
        h = ad.read_h5ad(str(pred_path), backed="r")
        try:
            col = "condition" if "condition" in h.obs.columns else h.obs.columns[0]
            return canon_set(h.obs[col].astype(str).unique())
        finally:
            try:
                h.file.close()
            except Exception:
                pass
    except Exception as e:
        log.warning("fold-align: could not read %s: %s", pred_path, e)
        return set()


def dl_predicted_bins(pred_path) -> Set[str]:
    """Lower-cased set of covariate (cell-type) values present in a DL prediction
    h5ad — used for the UnseenCell scoreability check (is our held-out cell present?)."""
    import anndata as ad
    h = ad.read_h5ad(str(pred_path), backed="r")
    try:
        col = ("covariate" if "covariate" in h.obs.columns else
               "cell_type" if "cell_type" in h.obs.columns else None)
        if col is None:
            return set()
        return {str(c).lower() for c in h.obs[col].astype(str).unique()}
    finally:
        try:
            h.file.close()
        except Exception:
            pass


# ===================================================================
# Per-model vocabulary (data/ref/dl_model_vocab/) — gene-set awareness
# ===================================================================
_VOCAB_DIR = DATA_DIR / "ref" / "dl_model_vocab"
_VOCAB_CACHE: Dict[Tuple[str, str], Optional[frozenset]] = {}


def load_model_vocab(model: str, dataset: str) -> Optional[frozenset]:
    """Gene-symbol vocabulary for a model, from data/ref/dl_model_vocab/.
    scGPT/GEARS are global ({model}.json); PRESAGE is per-dataset
    (presage_{dataset}.json). Returns None when no reference file exists (gate then
    degrades to fold-identity only — vocab classification is skipped)."""
    model = model.lower()
    key = (model, dataset)
    if key in _VOCAB_CACHE:
        return _VOCAB_CACHE[key]
    path = (_VOCAB_DIR / f"presage_{dataset}.json" if model == "presage"
            else _VOCAB_DIR / f"{model}.json")
    vocab: Optional[frozenset] = None
    if path.exists():
        try:
            vocab = frozenset(json.loads(path.read_text()).get("genes", []))
        except Exception as e:  # noqa: BLE001
            log.warning("load_model_vocab: failed to read %s: %s", path, e)
    _VOCAB_CACHE[key] = vocab
    return vocab


def _genes_of(pert_key: str) -> List[str]:
    """Target gene tokens of a canonical pert key (combos split on '+'/';')."""
    return [g for g in pert_key.replace(";", "+").split("+") if g]


def fold_alignment_verdict(
    *, scenario: str, model: str,
    declared: Optional[Set[str]] = None,
    h5ad_fold: Optional[Set[str]] = None,
    covered: Optional[Set[str]] = None,
    vocab: Optional[frozenset] = None,
    held_out_bins: Optional[Set[str]] = None,
    present_bins: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """Decide whether a DL fold's predictions may be adopted. Returns (ok, reason).

    Pert-distinguishable regimes (UnseenPert/Combo/Both/Pair) — all sets are
    CANONICAL pert keys (control already stripped by `canon_set`):
      * HARD FAIL if `declared ⊄ h5ad_fold` (the model's declared test set carries
        perturbations not in our fold → wrong fold / wrong partition).
      * HARD FAIL if `covered ⊄ declared` (the model emitted perts it never declared).
      * The declaration gap `h5ad_fold − declared` is tolerated ONLY when every gap
        pert's target gene is OUT-OF-VOCAB for this model (a genuine coverage limit);
        an IN-VOCAB gap pert is a fold-partition mismatch → HARD FAIL. (If no vocab
        ref is available the gap is reported as a warning, not failed.)
    UnseenCell (pert-degenerate — perts identical across folds): judged by
    SCOREABILITY — the source file must contain rows for our held-out test cell.
    """
    if scenario == "UnseenCell":
        if held_out_bins is None or present_bins is None:
            return False, "UnseenCell: missing bin info for scoreability check"
        missing = sorted(set(held_out_bins) - set(present_bins))
        if missing:
            return False, (f"UnseenCell: held-out test cell(s) {missing} ABSENT from the "
                           f"source prediction file → cell-axis unscoreable "
                           f"(orientation/leakage unverifiable). Refusing to adopt.")
        return True, f"UnseenCell: held-out cell(s) {sorted(held_out_bins)} present in source"

    if not declared:
        return False, f"{model}: no declared test set parsed from metadata"
    if h5ad_fold is None:
        return False, f"{model}: missing h5ad fold test set"

    contamination = sorted(declared - h5ad_fold)
    if contamination:
        return False, (f"{model}: declared test set has {len(contamination)} pert(s) NOT in "
                       f"our fold → wrong fold/partition: {contamination[:8]}")
    if covered is not None:
        extra = sorted(covered - declared)
        if extra:
            return False, (f"{model}: emitted {len(extra)} pert(s) outside the declared "
                           f"test set → contamination: {extra[:8]}")

    decl_gap = sorted(h5ad_fold - declared)
    in_vocab_gap, oov_gap = decl_gap, []
    if vocab is not None and decl_gap:
        oov_gap = [p for p in decl_gap if any(g not in vocab for g in _genes_of(p))]
        in_vocab_gap = [p for p in decl_gap if p not in set(oov_gap)]
        if in_vocab_gap:
            return False, (f"{model}: {len(in_vocab_gap)} fold pert(s) are IN this model's "
                           f"vocabulary yet absent from its declared test set → fold-partition "
                           f"mismatch (not a vocabulary limit): {in_vocab_gap[:8]}")

    reason = f"{model}: declared⊆fold ✓ (|declared|={len(declared)}, |fold|={len(h5ad_fold)})"
    if decl_gap:
        if vocab is not None:
            reason += f"; fold−declared={len(decl_gap)} all out-of-vocab ✓"
        else:
            reason += f"; fold−declared={len(decl_gap)} (no vocab ref — unclassified)"
    if covered is not None:
        emit_gap = sorted(declared - covered)
        if emit_gap:
            cls = ""
            if vocab is not None:
                oov = [p for p in emit_gap if any(g not in vocab for g in _genes_of(p))]
                cls = f" (oov={len(oov)}, in-vocab-dropped={len(emit_gap) - len(oov)})"
            reason += f"; declared−emitted={len(emit_gap)}{cls}"
    return True, reason
