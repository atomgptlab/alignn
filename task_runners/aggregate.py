"""Summarise a task's runs: mean +/- sd over seeds, and the LaTeX table.

Seeds are the whole point.  Across fifteen independently trained models the
match rate on the 103 JARVIS targets spanned 0.437-0.524, so a single run is
not evidence for a difference of a few percent, and this module refuses to
present one as though it were: every group is reported with its spread and
its n, and the paired comparisons print the change with both arms' spreads
next to it.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from task_runners import tasks as T

# AtomBench's own extraction of metrics.json, and the published baselines,
# rather than a second implementation that could drift from it.
sys.path.insert(0, str(T.REPO / "scripts" / "atombench"))
from collect_results import BASELINES, extract  # noqa: E402

#: label, key, decimals, lower-is-better
COLUMNS = [
    ("loss", "loss", 3, True),
    ("match", "match", 4, False),
    ("RMSD", "rmsd", 3, True),
    ("ccRMSD", "ccrmsd", 3, True),
    ("MAE abc", "abc", 3, True),
    ("MAE ang", "ang", 2, True),
    ("KLD", "kld", 4, True),
]

LATEX_HEADERS = {
    "loss": r"Denoising loss $\downarrow$",
    "match": r"Match rate $\uparrow$",
    "rmsd": r"Coordinate RMSD (\AA) $\downarrow$",
    "ccrmsd": r"ccRMSD $\downarrow$",
    "abc": r"Lattice MAE, $abc$ (\AA) $\downarrow$",
    "ang": r"Lattice MAE, angles ($^{\circ}$) $\downarrow$",
    "kld": r"KLD $\downarrow$",
}


def best_val_loss(history: Optional[Path]) -> Optional[float]:
    """Lowest validation denoising loss recorded during training."""
    if history is None or not history.exists():
        return None
    try:
        rows = json.loads(history.read_text())
    except json.JSONDecodeError:
        return None
    losses = [r["val"]["loss"] for r in rows if "val" in r]
    return min(losses) if losses else None


def unit_facts(unit: T.Unit) -> Dict:
    """Facts a metrics.json does not carry, read from the run directory.

    Parameter count and wall time make two of the manuscript's claims
    checkable -- "matched to within 1% on parameters" and "2.4x in time per
    training step" -- from artefacts the runner already writes.
    """
    facts: Dict = {}
    cfg = unit.rundir / "config.json"
    if cfg.exists():
        try:
            facts["params"] = json.loads(cfg.read_text()).get("n_parameters")
        except json.JSONDecodeError:
            pass
    marker = unit.rundir / ".stages" / "train.json"
    if marker.exists():
        try:
            facts["train_s"] = json.loads(marker.read_text()).get("elapsed_s")
        except json.JSONDecodeError:
            pass
    for name, keys in (
        ("split_meta.json", ("n_train", "n_val", "n_test")),
        ("leakage.json", ("fraction", "n_leaked", "n_test")),
    ):
        path = unit.rundir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for key in keys:
            if key in data:
                facts[key] = data[key]
        if "fraction" in data:
            facts["leak_fraction"] = data["fraction"]
    return facts


def collect(units: Sequence[T.Unit], variant: str) -> Dict[str, List[Dict]]:
    """Group the units' results by aggregation label.

    A unit whose metrics are not keyed by ``variant`` (the tolerance sweep,
    the leakage filter) contributes one row per key it does have, so those
    tasks read as several groups of one rather than needing a special case.
    """
    groups: Dict[str, List[Dict]] = {}
    for unit in units:
        loss = best_val_loss(unit.history)
        if variant in unit.metrics:
            keyed = {unit.group or unit.name: unit.metrics[variant]}
        elif unit.metrics:
            prefix = f"{unit.group}:" if len(units) > 1 else ""
            keyed = {f"{prefix}{k}": v for k, v in unit.metrics.items()}
        else:
            keyed = {unit.group or unit.name: None}
        facts = unit_facts(unit)
        for label, path in keyed.items():
            row = {"unit": unit.name, "seed": unit.seed, "loss": loss}
            row.update(facts)
            if path is not None and Path(path).exists():
                row.update(extract(Path(path)))
            elif path is not None:
                row["missing"] = str(path)
            groups.setdefault(label, []).append(row)
    return groups


def stat(rows: Sequence[Dict], key: str):
    """(mean, sd, n) over the rows that actually have this metric."""
    vals = [
        r[key] for r in rows if r.get(key) is not None and not _nan(r.get(key))
    ]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return statistics.fmean(vals), statistics.stdev(vals), len(vals)


def _nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def cell(mean, sd, dp: int) -> str:
    if mean is None:
        return "-"
    if sd is None:
        return f"{mean:.{dp}f}"
    return f"{mean:.{dp}f}+-{sd:.{dp}f}"


def welch_p(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Two-sided Welch p-value, if scipy is around to give one."""
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy import stats
    except ImportError:
        return None
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def values(rows: Sequence[Dict], key: str) -> List[float]:
    return [
        r[key] for r in rows if r.get(key) is not None and not _nan(r.get(key))
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(groups: Dict[str, List[Dict]]) -> None:
    label_w = max([len(k) for k in groups] + [12]) + 2
    header = (
        "group".ljust(label_w)
        + "".join(f"{lab:>17}" for lab, _, _, _ in COLUMNS)
        + f"{'n':>4}"
    )
    print(header)
    print("-" * len(header))
    for label, rows in groups.items():
        line = label.ljust(label_w)
        n_max = 0
        for _, key, dp, _ in COLUMNS:
            mean, sd, n = stat(rows, key)
            n_max = max(n_max, n)
            line += f"{cell(mean, sd, dp):>17}"
        print(line + f"{n_max:>4}")


def print_notes(units: Sequence[T.Unit]) -> None:
    """Headline numbers a metrics.json does not carry.

    Currently just the leakage fraction: how many test targets have a
    structure-matcher-identical counterpart in the pretraining corpus, and so
    are reachable by recall rather than by generation.
    """
    for unit in units:
        report = unit.rundir / "leakage.json"
        if not report.exists():
            continue
        try:
            data = json.loads(report.read_text())
        except json.JSONDecodeError:
            continue
        print(
            f"  {unit.name}: {data['n_leaked']}/{data['n_test']} test targets"
            f" ({data['fraction']:.1%}) are reachable by recall"
        )
    print()


def print_legend(task: T.Task, groups: Dict[str, List[Dict]]) -> None:
    """Spell out the short group labels, so the table can stay narrow."""
    entries = [(k, v) for k, v in task.legend.items() if k in groups]
    if not entries:
        return
    width = max(len(k) for k, _ in entries)
    print()
    for label, text in entries:
        print(f"  {label.ljust(width)}  {text}")


def print_runs(groups: Dict[str, List[Dict]]) -> None:
    """Per-run rows: an outlier seed should be visible, not averaged away."""
    print("\nindividual runs")
    for label, rows in groups.items():
        for row in rows:
            if row.get("missing"):
                print(f"  {label:<34} {row['unit']:<24} not scored yet")
                continue
            bits = []
            for lab, key, dp, _ in COLUMNS:
                val = row.get(key)
                bits.append(
                    f"{lab} {val:.{dp}f}"
                    if val is not None and not _nan(val)
                    else f"{lab} -"
                )
            print(f"  {label:<34} {row['unit']:<24} " + "  ".join(bits))


def _hms(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m"


def print_cost(groups: Dict[str, List[Dict]]) -> None:
    """Parameters and training wall time per arm.

    The time ratio equals the per-step ratio only when the arms ran the same
    number of epochs on the same hardware, which every task here arranges and
    a job array does not guarantee -- so it is labelled as wall time, not as
    cost per step.
    """
    rows = [
        (label, stat(r, "params"), stat(r, "train_s"))
        for label, r in groups.items()
    ]
    rows = [r for r in rows if r[1][0] is not None or r[2][0] is not None]
    if not rows:
        return
    ref = next((r[2][0] for r in rows if r[2][0]), None)
    width = max(len(r[0]) for r in rows) + 2
    print("\ntraining cost")
    print(
        "  " + "arm".ljust(width) + f"{'params':>10}{'wall time':>14}"
        f"{'x vs first':>12}"
    )
    for label, (params, _, _), (secs, sd, _) in rows:
        par = "-" if params is None else f"{params / 1e6:.2f} M"
        if secs is None:
            time_txt, ratio = "-", "-"
        else:
            time_txt = _hms(secs) + (f" +-{_hms(sd)}" if sd else "")
            ratio = f"{secs / ref:.2f}" if ref else "-"
        print(f"  {label.ljust(width)}{par:>10}{time_txt:>14}{ratio:>12}")


def print_comparisons(task: T.Task, groups: Dict[str, List[Dict]]) -> None:
    if not task.comparisons:
        return
    print("\ncomparisons  (change of the second arm relative to the first)")
    for question, (ref, test) in task.comparisons.items():
        if ref not in groups or test not in groups:
            missing = [g for g in (ref, test) if g not in groups]
            print(f"\n  {question}: not run ({', '.join(missing)})")
            continue
        print(f"\n  {question}")
        print(f"    {ref}  ->  {test}")
        for lab, key, dp, lower in COLUMNS:
            m_ref, s_ref, n_ref = stat(groups[ref], key)
            m_test, s_test, n_test = stat(groups[test], key)
            if m_ref is None or m_test is None:
                continue
            change = (
                (m_test - m_ref) / m_ref * 100.0 if m_ref else float("nan")
            )
            direction = "better" if (change < 0) == lower else "worse"
            if abs(change) < 0.05:
                direction = "unchanged"
            p = welch_p(values(groups[ref], key), values(groups[test], key))
            # A change smaller than the arms' own spread is not a result.
            spread = max(s_ref or 0.0, s_test or 0.0)
            noise = (
                "  (within noise)"
                if spread and abs(m_test - m_ref) < spread
                else ""
            )
            p_txt = f"  p={p:.3f}" if p is not None else ""
            print(
                f"      {lab:<9} {cell(m_ref, s_ref, dp):>17} -> "
                f"{cell(m_test, s_test, dp):>17}"
                f"  {change:+6.1f}%  {direction}{p_txt}{noise}"
            )


def print_baselines(task: T.Task, groups: Dict[str, List[Dict]]) -> None:
    if not task.baselines:
        return
    base = BASELINES[task.baselines]
    print(f"\npublished AtomBench baselines ({task.baselines})")
    label_w = max([len(k) for k in base] + [len(k) + 7 for k in groups]) + 2
    cols = [c for c in COLUMNS if c[1] != "loss"]
    print(
        "model".ljust(label_w) + "".join(f"{lab:>10}" for lab, _, _, _ in cols)
    )
    for name, row in base.items():
        line = name.ljust(label_w)
        for _, key, dp, _ in cols:
            line += f"{row.get(key, float('nan')):>10.{dp}f}"
        print(line)
    for label, rows in groups.items():
        line = f"{label} (ours)".ljust(label_w)
        for _, key, dp, _ in cols:
            mean, _, _ = stat(rows, key)
            line += ("-" if mean is None else f"{mean:.{dp}f}").rjust(10)
        print(line)
        # The manuscript also quotes the best individual run; name it here
        # rather than letting a reader assume the column-wise best is one run.
        scored = [r for r in rows if r.get("match") is not None]
        if len(scored) > 1:
            best = max(scored, key=lambda r: r["match"])
            line = f"  best run ({best['unit']})".ljust(label_w)
            for _, key, dp, _ in cols:
                val = best.get(key)
                line += (
                    "-" if val is None or _nan(val) else f"({val:.{dp}f})"
                ).rjust(10)
            print(line)


# ---------------------------------------------------------------------------
# Claim resolution, for `verify`
# ---------------------------------------------------------------------------


def resolve(rows: Sequence[Dict], metric: str):
    """(value, sd, n) for a metric name, including the derived ones.

    ``match_min`` / ``match_max`` give the seed spread; ``bestrun_<m>`` gives
    metric *m* as measured in the run with the highest match rate, which is
    what the manuscript's parenthesised "best individual run" row is.
    """
    if metric.startswith("bestrun_"):
        key = metric[len("bestrun_") :]
        scored = [r for r in rows if r.get("match") is not None]
        if not scored:
            return None, None, 0
        best = max(scored, key=lambda r: r["match"])
        value = best.get(key)
        return (None, None, 0) if _nan(value) else (value, None, 1)
    if metric in ("match_min", "match_max"):
        vals = values(rows, "match")
        if not vals:
            return None, None, 0
        return (
            (min(vals) if metric.endswith("min") else max(vals)),
            None,
            len(vals),
        )
    return stat(rows, metric)


def check(
    task: T.Task,
    units: Sequence[T.Unit],
    claim,
) -> Dict:
    """Measure one claim against the runs on disk."""
    groups = collect(units, claim.variant)
    rows = groups.get(claim.group or next(iter(groups), ""), [])
    value, sd, n = resolve(rows, claim.metric)
    if claim.ref_group:
        ref_rows = groups.get(claim.ref_group, [])
        ref, _, ref_n = resolve(ref_rows, claim.metric)
        if value is None or not ref:
            value, sd, n = None, None, min(n, ref_n)
        else:
            value, sd, n = value / ref, None, min(n, ref_n)

    out = {"measured": value, "sd": sd, "n": n, "status": "not run"}
    if value is None:
        return out
    published = claim.published
    rel = (value - published) / published if published else float("inf")
    out["rel"] = rel
    if abs(rel) <= claim.tol:
        out["status"] = "ok"
    elif sd and abs(value - published) <= sd:
        # Inside one standard deviation of the published number is agreement
        # at this sample size, whatever the relative gap looks like.
        out["status"] = "within sd"
    else:
        out["status"] = "differs"
    return out


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def latex_ablation(task: T.Task, groups: Dict[str, List[Dict]]) -> str:
    """Metrics down the side, arms across: the shape of Table 3."""
    labels = list(groups)
    lines = [
        r"\begin{tabular}{l" + "c" * (len(labels) + 1) + "}",
        r"\hline",
        "Metric & " + " & ".join(labels) + r" & Change \\",
        r"\hline",
    ]
    # The Change column is the task's own comparison, so it reads the same
    # way as the printed one: the second arm relative to the first, not
    # whichever arm happens to be leftmost in the table.
    ref, test = next(iter(task.comparisons.values()), (labels[0], labels[-1]))
    for lab, key, dp, lower in COLUMNS:
        cells = []
        means = {}
        for label in labels:
            mean, sd, _ = stat(groups[label], key)
            means[label] = mean
            cells.append("-" if mean is None else _tex_cell(mean, sd, dp))
        if means.get(ref) and means.get(test) is not None:
            change = (means[test] - means[ref]) / means[ref] * 100
            change_txt = f"${change:+.0f}\\%$"
        else:
            change_txt = "-"
        lines.append(
            f"{LATEX_HEADERS[key]} & "
            + " & ".join(cells)
            + f" & {change_txt} "
            + r"\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def latex_bench(task: T.Task, groups: Dict[str, List[Dict]]) -> str:
    """Models down the side, metrics across: the shape of Table 4."""
    cols = [c for c in COLUMNS if c[1] != "loss"]
    lines = [
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\hline",
        "Model & "
        + " & ".join(LATEX_HEADERS[k] for _, k, _, _ in cols)
        + r" \\",
        r"\hline",
    ]
    for name, row in BASELINES[task.baselines].items():
        cells = [f"${row[k]:.{dp}f}$" for _, k, dp, _ in cols]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    for label, rows in groups.items():
        cells = []
        for _, key, dp, _ in cols:
            mean, sd, _ = stat(rows, key)
            cells.append("-" if mean is None else _tex_cell(mean, sd, dp))
        lines.append(f"ALIGNN-CSP ({label}) & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def _tex_cell(mean: float, sd: Optional[float], dp: int) -> str:
    if sd is None:
        return f"${mean:.{dp}f}$"
    return f"${mean:.{dp}f}\\pm{sd:.{dp}f}$"


def latex_generic(groups: Dict[str, List[Dict]]) -> str:
    lines = [
        r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}",
        r"\hline",
        "Run & "
        + " & ".join(LATEX_HEADERS[k] for _, k, _, _ in COLUMNS)
        + r" \\",
        r"\hline",
    ]
    for label, rows in groups.items():
        cells = []
        for _, key, dp, _ in COLUMNS:
            mean, sd, _ = stat(rows, key)
            cells.append("-" if mean is None else _tex_cell(mean, sd, dp))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def report(
    task: T.Task,
    units: Sequence[T.Unit],
    *,
    variant: str = "sym",
    latex: bool = False,
) -> int:
    groups = collect(units, variant)
    scored = sum(
        1
        for rows in groups.values()
        for r in rows
        if r.get("match") is not None or r.get("loss") is not None
    )
    print(f"\n{task.name}: {task.summary}")
    print(f"reproduces: {task.reproduces}")
    print(
        f"variant: {variant}   groups: {len(groups)}   with results: "
        f"{scored}\n"
    )
    if not any(u.metrics or u.history for u in units):
        # A data-preparation task has nothing to average; say so and exit
        # clean, so the dependent aggregation job is not a red herring.
        print(
            "This task produces inputs, not metrics -- nothing to "
            "aggregate."
        )
        return 0
    if not scored:
        print("Nothing scored yet. Run the task first:")
        print(f"  python task_runners/run_task.py {task.name}")
        return 1

    print_notes(units)
    print_table(groups)
    print_legend(task, groups)
    print_runs(groups)
    print_cost(groups)
    print_comparisons(task, groups)
    print_baselines(task, groups)

    if latex:
        print("\n% ---- LaTeX ----")
        if task.baselines:
            print(latex_bench(task, groups))
        elif len(groups) == 2 and task.comparisons:
            print(latex_ablation(task, groups))
        else:
            print(latex_generic(groups))
    print()
    return 0
