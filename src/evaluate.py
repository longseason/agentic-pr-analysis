"""Multi-seed comparison, feature-group ablation, temporal split and calibration for the merge models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import new_model

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]


def feature_groups(columns):
    groups = {
        "agent": [c for c in columns if c.startswith("agent_")],
        "task_type": [c for c in columns if c.startswith("task_type_")],
        "repository": [c for c in columns if c.startswith("lang_")]
        + ["log_stars", "log_forks", "is_forked", "has_license"],
        "author_history": ["log_user_prior_prs"],
    }
    assert sum(len(v) for v in groups.values()) == len(columns)
    return groups


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(
        abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
        for b in range(bins)
        if (idx == b).any()
    )


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y, p),
        "macro_f1": f1_score(y, pred, average="macro"),
        "acc": accuracy_score(y, pred),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
    }


def models():
    return {
        "logreg": lambda: LogisticRegression(max_iter=1000),
        "gbt": lambda: HistGradientBoostingClassifier(random_state=0),
    }


def fit_predict(make, X_tr, y_tr, X_te):
    return make().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def grouped_splits(X, y, groups):
    for seed in SEEDS:
        tr, te = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                X, y, groups
            )
        )
        yield seed, tr, te


def main():
    OUT.mkdir(exist_ok=True)
    new_model.ensure_prs()
    df = new_model.load_prs()
    X, y, groups = new_model.build_features(df)
    fg = feature_groups(list(X.columns))
    base_cols = fg["agent"] + fg["task_type"]
    y_np = y.to_numpy()
    rows = []

    for seed, tr, te in grouped_splits(X, y, groups):
        y_te = y_np[te]
        rows.append(
            {
                "seed": seed,
                "model": "majority",
                "features": "none",
                **scores(y_te, np.full(len(te), y_np[tr].mean())),
            }
        )
        for name, make in models().items():
            for feats, cols in [("agent+task", base_cols), ("full", list(X.columns))]:
                p = fit_predict(make, X.iloc[tr][cols], y_np[tr], X.iloc[te][cols])
                rows.append(
                    {"seed": seed, "model": name, "features": feats, **scores(y_te, p)}
                )
        # leave-one-group-out and single-group ablation of the full logistic model
        for g, cols in fg.items():
            keep = [c for c in X.columns if c not in cols]
            p = fit_predict(
                models()["logreg"], X.iloc[tr][keep], y_np[tr], X.iloc[te][keep]
            )
            rows.append(
                {
                    "seed": seed,
                    "model": "logreg",
                    "features": f"-{g}",
                    **scores(y_te, p),
                }
            )
            p = fit_predict(
                models()["logreg"], X.iloc[tr][cols], y_np[tr], X.iloc[te][cols]
            )
            rows.append(
                {
                    "seed": seed,
                    "model": "logreg",
                    "features": f"only {g}",
                    **scores(y_te, p),
                }
            )
        print("seed", seed, "done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "grouped_splits.csv", index=False)
    summary = (
        res.groupby(["model", "features"])
        .agg(["mean", "std"])
        .drop(columns="seed")
        .round(3)
    )
    summary.to_csv(OUT / "grouped_summary.csv")
    print(summary.to_string())

    # temporal split: train on the earliest 80% of PRs by creation time, test on the rest
    order = df["created_at"].sort_values(kind="stable").index
    cut = int(0.8 * len(order))
    tr, te = order[:cut].to_numpy(), order[cut:].to_numpy()
    temporal = {
        "cutoff": str(df.loc[order[cut], "created_at"]),
        "test_merge_rate": float(y_np[te].mean()),
    }
    for name, make in models().items():
        for feats, cols in [("agent+task", base_cols), ("full", list(X.columns))]:
            p = fit_predict(make, X.iloc[tr][cols], y_np[tr], X.iloc[te][cols])
            temporal[f"{name} {feats}"] = {
                k: round(v, 3) for k, v in scores(y_np[te], p).items()
            }
    print(json.dumps(temporal, indent=1))

    # paired bootstrap of the AUROC gain from context features on the seed-0 split
    _, tr, te = next(grouped_splits(X, y, groups))
    p_base = fit_predict(
        models()["logreg"], X.iloc[tr][base_cols], y_np[tr], X.iloc[te][base_cols]
    )
    p_full = fit_predict(models()["logreg"], X.iloc[tr], y_np[tr], X.iloc[te])
    rng = np.random.default_rng(0)
    test_groups = groups.to_numpy()[te]
    uniq = np.unique(test_groups)
    members = pd.Series(np.arange(len(te))).groupby(test_groups).apply(np.array)
    diffs = []
    for _ in range(1000):
        # resample repositories, not PRs, because PRs in one repository are correlated
        idx = np.concatenate(members.loc[rng.choice(uniq, len(uniq))].to_numpy())
        diffs.append(
            roc_auc_score(y_np[te][idx], p_full[idx])
            - roc_auc_score(y_np[te][idx], p_base[idx])
        )
    boot = {
        "delta_auroc": round(
            roc_auc_score(y_np[te], p_full) - roc_auc_score(y_np[te], p_base), 3
        ),
        "ci95": [round(float(q), 3) for q in np.percentile(diffs, [2.5, 97.5])],
    }
    print("bootstrap", boot)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bins = np.linspace(0, 1, 11)
    for label, p in [("agent + task type", p_base), ("full context", p_full)]:
        idx = np.digitize(p, bins[1:-1])
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 50]
        ys = [y_np[te][idx == b].mean() for b in range(10) if (idx == b).sum() >= 50]
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_xlabel("Mean predicted merge probability")
    ax.set_ylabel("Observed merge rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=200)

    rates, counts = new_model.merge_rates(df)
    rates.round(3).to_csv(OUT / "merge_rates.csv")
    counts.to_csv(OUT / "merge_counts.csv")
    new_model.plot_merge_rates(rates, path=OUT / "merge_rates.png")

    json.dump(
        {"temporal": temporal, "bootstrap": boot},
        open(OUT / "extra.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
