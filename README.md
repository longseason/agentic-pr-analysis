# agentic-pr-analysis

Analysis of AI-agent pull requests from [AIDev](https://huggingface.co/datasets/hao-li/AIDev), release v5.

## Pipeline

```bash
pip install -r requirements.txt
cd src && python data.py
```

`src/data.py` downloads v5 `pull_request`, `pr_reviews`, `repository`, `pr_timeline` into `data/raw/`,
adds task types, and writes `data/processed/prs.parquet` plus the other tables.

## Task types

AIDev dropped the `pr_task_type` table after v3. Task types for v5 come from a BERT classifier
trained on the v3 labels: [Hydragon123/aidev-task-type-bert](https://huggingface.co/Hydragon123/aidev-task-type-bert).

Columns added to `prs.parquet`:

| column                 | meaning                                                                         |
| ---------------------- | ------------------------------------------------------------------------------- |
| `task_type`            | `build, chore, ci, docs, feat, fix, other, perf, refactor, revert, style, test` |
| `task_type_confidence` | softmax max; 1.0 for v3 labels                                                  |
| `task_type_source`     | `v3_label` (PR labeled in v3, label kept) or `bert`                             |

`data.py` reads cached labels from `data/processed/pull_request_task_type_v5.parquet`.
On a cache miss it runs the model. Run that step on a GPU (Colab), not locally.

```bash
cd src
python task_type_classifier.py label                         # v5 pull_request -> data/processed/pull_request_task_type_v5.parquet
python task_type_classifier.py label --table all_pull_request
python task_type_classifier.py train --push                  # retrain from v3 labels, upload to HF
```

Validation on 3,360 held-out v3 PRs: accuracy 0.894, weighted F1 0.893, macro F1 0.627.
`style` (F1 0.43) and `build` (0.52) are weak. `other` and `revert` are never predicted (31 and 16 training rows).
Full per-class table on the model card.

## Labeling time

Measured throughput: 192 PRs/s on a Colab T4 (fp16, batch 256, max 256 tokens).
CPU figure from a 500-PR run on a Colab CPU VM (2 vCPU): at most 2.6 PRs/s.

| table                 |       PRs |             T4 |
| --------------------- | --------: | -------------: |
| v5 `pull_request`     |   361,296 |         31 min |
| v5 `all_pull_request` | 7,685,281 | ~11.1 h (est.) |

Labeling is chunked automatically. Each parquet row group (200K PRs, ~17 min on T4) is streamed from HF,
labeled, and saved to `<out>_chunks/`. A killed run resumes on rerun with the same `--out`.
Chunks merge into `<out>` and the chunk dir is removed when all are done.
## Clustering pull requests

After generating `data/processed/prs.parquet` with `src/data.py`, run:

```powershell
python src\clustering.py
```

The script samples at most 2,000 rows from the parquet file in batches to keep
memory use bounded. Adjust the sample or cluster count with `--sample-size` and
`--clusters`.

It converts pull-request titles and bodies to TF-IDF features, reduces those
features with PCA, and runs K-means and hierarchical (Ward) clustering. The
printed cluster summaries compare task type, agent, and merge rate after fitting;
none of those columns, including `merged`, is used as a clustering feature.
PCA is dimensionality reduction and visualization, not a clustering algorithm.
Clustering is exploratory: it does not train a model to predict an agent or
whether a pull request will merge. Use a supervised classifier and held-out
evaluation for that prediction goal. The dataset's `agent` field identifies the
agent/tool category and may not identify the underlying language model.
