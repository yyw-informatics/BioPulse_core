"""Deterministic Open Problems control outputs for benchmark packs.

OP normalizes leaderboard scores against two control_methods per task — a *random* lower anchor and a
*true/oracle* upper anchor — so a raw metric becomes ``(raw - random) / (oracle - random)``. We
reproduce those controls here and store them under ``baselines/baseline_outputs/{random,oracle}.h5ad``
so the scorer can report the normalized score alongside the raw one (see
``scorers.common.normalize_to_controls`` / ``load_control_outputs``).

Reproduces OP's ``src/control_methods`` with a seeded RNG (OP seeds the SVG ranking but not
label_projection's random labels), so the stored controls — and the normalized scores derived from
them — are reproducible. Control files live outside ``public/`` and are never shown to the agent.
label_projection and spatially_variable_genes produce local controls; the other tasks report raw
scores only.

CONTROL_SEED is fixed; change it only deliberately, as it shifts every pack's normalized scores.
"""

from __future__ import annotations

from pathlib import Path

CONTROL_SEED = 0


def write_controls(pack_dir: Path | str) -> list[Path]:
    """Generate random/oracle control outputs for ``pack_dir``.

    Tasks without local controls return an empty list. The task_type -> generator mapping lives in the
    task registry (each TaskRecord's ``controls`` field), so adding a task wires its controls in one place.
    """
    from biopulse.tasks.registry import get  # local import: registry references this module's generators

    pack = Path(pack_dir)
    record = get(_read_task_type(pack))
    return record.controls(pack) if record.controls else []


def _read_task_type(pack: Path) -> str:
    from biopulse.scorers.common import read_task_yaml

    return str(read_task_yaml(pack / "task.yaml").get("task_type", ""))


def _controls_dir(pack: Path) -> Path:
    out = pack / "baselines" / "baseline_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_label_projection_controls(pack: Path) -> list[Path]:
    import anndata as ad
    import numpy as np
    import pandas as pd

    solution = ad.read_h5ad(pack / "hidden" / "ground_truth" / "solution.h5ad")
    label_col = "label" if "label" in solution.obs else "cell_type"
    true_labels = solution.obs[label_col].astype(str)
    index = solution.obs_names

    # random_labels: sample from the training label distribution (fall back to the solution's own
    # distribution if the public train carries no labels), seeded for reproducibility.
    train_path = pack / "public" / "input" / "train.h5ad"
    dist_source = true_labels
    if train_path.exists():
        train = ad.read_h5ad(train_path)
        train_col = "label" if "label" in train.obs else "cell_type" if "cell_type" in train.obs else None
        if train_col is not None:
            dist_source = train.obs[train_col].astype(str)
    distribution = dist_source.value_counts()
    distribution = distribution / distribution.sum()
    rng = np.random.RandomState(CONTROL_SEED)
    random_pred = rng.choice(distribution.index.to_numpy(), size=len(index), replace=True, p=distribution.to_numpy())

    out_dir = _controls_dir(pack)
    written: list[Path] = []
    for name, preds in (("random", random_pred), ("oracle", true_labels.to_numpy())):
        obj = ad.AnnData(obs=pd.DataFrame({"label_pred": preds}, index=index), uns={"method_id": f"control_{name}"})
        path = out_dir / f"{name}.h5ad"
        obj.write_h5ad(path)
        written.append(path)
    return written


def make_svg_controls(pack: Path) -> list[Path]:
    import anndata as ad
    import numpy as np

    solution = ad.read_h5ad(pack / "hidden" / "ground_truth" / "solution.h5ad")
    sv = solution.var
    feature_id = sv["feature_id"].to_numpy() if "feature_id" in sv else solution.var_names.to_numpy()
    true_scores = sv["true_spatial_var_score"].to_numpy().astype(float)

    rng = np.random.RandomState(CONTROL_SEED)
    random_scores = rng.rand(len(feature_id))

    out_dir = _controls_dir(pack)
    written: list[Path] = []
    for name, scores in (("random", random_scores), ("oracle", true_scores)):
        var = _svg_var(feature_id, scores)
        obj = ad.AnnData(var=var, uns={"method_id": f"control_{name}"})
        obj.var_names = [str(value) for value in feature_id]
        path = out_dir / f"{name}.h5ad"
        obj.write_h5ad(path)
        written.append(path)
    return written


def _svg_var(feature_id, scores):
    import pandas as pd

    return pd.DataFrame(
        {"feature_id": [str(value) for value in feature_id], "pred_spatial_var_score": scores},
        index=[str(value) for value in feature_id],
    )
