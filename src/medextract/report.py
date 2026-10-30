"""Turn the MLflow experiment into the markdown table the README carries.

Keeping this a script means the table cannot drift from the tracking store --
regenerate it after a sweep instead of transcribing numbers by hand.

    python -m medextract.report --experiment medextract-ner > docs/sweep_table.md
"""

from __future__ import annotations

import argparse

from medextract.config import mlflow_tracking_uri

PARAM_COLUMNS = [
    ("base_model", "base model"),
    ("learning_rate", "lr"),
    ("batch_size", "batch"),
    ("epochs", "epochs"),
    ("seed", "seed"),
    ("label_all_subwords", "all subwords"),
]

METRIC_COLUMNS = [
    ("metrics.eval_precision", "val P"),
    ("metrics.eval_recall", "val R"),
    ("metrics.eval_f1", "val F1"),
    ("metrics.test_f1", "test F1"),
]


def per_type_columns(columns: list[str]) -> list[tuple[str, str]]:
    """Per-type F1 columns, present only when the experiment has more than one type.

    `entity_scores` logs `<Type>_f1` for every type a corpus annotates. On a
    single-type corpus that is the micro F1 again under another name, so it is
    left out; the moment a second type appears the aggregate stops saying
    which type moved, and the columns earn their place.
    """
    # `macro_f1` is an aggregate logged alongside the per-type scores, not a type.
    types = sorted(
        {
            c.removeprefix("metrics.eval_").removesuffix("_f1")
            for c in columns
            if c.startswith("metrics.eval_") and c.endswith("_f1") and c != "metrics.eval_f1"
        }
        - {"macro"}
    )
    if len(types) < 2:
        return []
    out: list[tuple[str, str]] = []
    for etype in types:
        out.append((f"metrics.eval_{etype}_f1", f"val F1 {etype}"))
    for etype in types:
        out.append((f"metrics.test_{etype}_f1", f"test F1 {etype}"))
    return out


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        return f"{value:.4f}" if abs(value) < 1 else f"{value:g}"
    text = str(value)
    return text.rsplit("/", 1)[-1] if "/" in text else text


def build_table(
    experiment: str,
    tracking_uri: str | None = None,
    sort_by: str = "metrics.eval_f1",
) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or mlflow_tracking_uri())
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"No MLflow experiment named {experiment!r}. Run a sweep first.")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        raise SystemExit(f"Experiment {experiment!r} has no runs yet.")

    if sort_by in runs.columns:
        runs = runs.sort_values(sort_by, ascending=False)

    return render_table(runs)


def render_table(runs) -> str:
    """Markdown from a `mlflow.search_runs` frame; separate so it is testable offline."""
    param_columns = list(PARAM_COLUMNS)
    if "params.keep_types" in runs.columns and runs["params.keep_types"].nunique() > 1:
        param_columns.append(("keep_types", "types"))
    metric_columns = list(METRIC_COLUMNS) + per_type_columns(list(runs.columns))

    headers = ["run"] + [h for _, h in param_columns] + [h for _, h in metric_columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for _, row in runs.iterrows():
        cells = [_fmt(row.get("tags.mlflow.runName"))]
        cells += [_fmt(row.get(f"params.{key}")) for key, _ in param_columns]
        cells += [_fmt(row.get(key)) for key, _ in metric_columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the MLflow sweep as a markdown table.")
    p.add_argument("--experiment", default="medextract-ner")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--sort-by", default="metrics.eval_f1")
    ns = p.parse_args(argv)
    print(build_table(ns.experiment, ns.tracking_uri, ns.sort_by))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
