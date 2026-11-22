# agentic-pr-analysis

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
