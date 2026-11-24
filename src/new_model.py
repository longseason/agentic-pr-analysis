from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data

ROOT = Path(__file__).resolve().parent.parent
PRS_PATH = ROOT / "data" / "processed" / "prs.parquet"
REPO_PATH = ROOT / "data" / "processed" / "repository.parquet"
LABELS_PATH = ROOT / "data" / "processed" / "pull_request_task_type_v5.parquet"

def ensure_prs():
    if PRS_PATH.exists():
        return
    if not LABELS_PATH.exists():
        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download("Hydragon123/aidev-task-type-v5", LABELS_PATH.name, repo_type="dataset")
        shutil.copy(path, LABELS_PATH)
    data.clean_data(data.load_data())


def load_prs():
    cols = ["id", "repo_id", "agent", "task_type", "task_type_confidence", "merged",
            "user", "created_at"]
    df = pd.read_parquet(PRS_PATH, columns=cols)
    df = df.dropna(subset=["repo_id", "agent", "task_type"])

    repos = pd.read_parquet(REPO_PATH, columns=["id", "stars", "forks", "language", "is_forked", "license"])
    repos = repos.rename(columns={"id": "repo_id"})
    repos["repo_id"] = repos["repo_id"].astype("Int64")
    df["repo_id"] = df["repo_id"].astype("Int64")

    return df.merge(repos, on="repo_id", how="left").reset_index(drop=True)

def build_features(df):
    X = pd.get_dummies(df[["agent", "task_type"]], dtype=int)

    X["log_stars"] = np.log1p(df["stars"].astype(float).fillna(0))
    X["log_forks"] = np.log1p(df["forks"].astype(float).fillna(0))
    X["is_forked"] = df["is_forked"].astype("boolean").fillna(False).astype(int)
    X["has_license"] = df["license"].notna().astype(int)

    lang = df["language"].astype(object)
    top_langs = lang.value_counts().index[:10]
    lang = lang.where(lang.isin(top_langs), "Other")
    X = X.join(pd.get_dummies(lang, prefix="lang", dtype=int))

    X["log_user_prior_prs"] = np.log1p(df.groupby("user")["created_at"].rank(method="first") - 1)

    y = df["merged"].astype(int)
    groups = df["repo_id"]
    return X, y, groups

def split(X, y, groups, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx])

def fake_prs(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "repo_id": rng.integers(0, 300, n),
        "agent": rng.choice(["Codex", "Copilot", "Claude Code", "Cursor", "Jules", "Devin"], n),
        "task_type": rng.choice(["fix", "feat", "docs", "chore", "refactor", "test"], n),
        "merged": rng.random(n) < 0.6,
    })

def cross_validate(X_train, y_train, g_train, folds=5):
    model = LogisticRegression(max_iter=1000)
    cv = GroupKFold(n_splits=folds)
    scores = cross_val_score(model, X_train, y_train, groups=g_train, cv=cv, scoring="roc_auc")
    return scores

def final_eval(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("test roc_auc:", roc_auc_score(y_test, probs))
    print("test accuracy:", accuracy_score(y_test, preds))
    print("baseline accuracy:", max(y_test.mean(), 1 - y_test.mean()))
    print(confusion_matrix(y_test, preds))

    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
    print(coefs)

    check = pd.DataFrame({"predicted": probs, "actual": y_test.to_numpy()})
    check["bucket"] = pd.cut(check["predicted"], bins=np.arange(0, 1.05, 0.1))
    print(check.groupby("bucket", observed=True).agg(n=("actual", "size"),
                                                     predicted=("predicted", "mean"),
                                                     actual=("actual", "mean")).round(3))

    return model, coefs

def merge_rates(df, min_prs=50):
    rates = df.pivot_table(index="agent", columns="task_type", values="merged", aggfunc="mean")
    counts = df.pivot_table(index="agent", columns="task_type", values="merged", aggfunc="size")
    rates = rates.where(counts >= min_prs)
    return rates, counts

def plot_merge_rates(rates, path=ROOT / "merge_rates.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(rates.values, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(rates.columns)), rates.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(rates.index)), rates.index)

    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            v = rates.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v > 0.6 else "white", fontsize=8)

    fig.colorbar(im, ax=ax, label="merge rate")
    ax.set_title("Merge rate by agent and task type")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()

if __name__ == "__main__":
    ensure_prs()
    df = load_prs()

    print(df.shape, "merge rate:", round(df["merged"].mean(), 3))

    X, y, groups = build_features(df)
    X_train, X_test, y_train, y_test, g_train = split(X, y, groups)
    print("train/test:", len(X_train), len(X_test))

    scores = cross_validate(X_train, y_train, g_train)
    print("cv roc_auc:", scores.round(3), "mean", scores.mean().round(3), "std", scores.std().round(3))

    model, coefs = final_eval(X_train, X_test, y_train, y_test)

    rates, counts = merge_rates(df)
    print(rates.round(2))
    plot_merge_rates(rates)