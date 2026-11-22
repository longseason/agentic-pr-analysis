import subprocess
import sys
from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / 'data' / 'raw'
PROCESSED_PATH = ROOT / 'data' / 'processed'

REVISION = 'v5'
TABLES = ['pull_request', 'pr_reviews', 'repository', 'pr_timeline']
TASK_TYPE_PATH = PROCESSED_PATH / f'pull_request_task_type_{REVISION}.parquet'

def load_data():
    # load_dataset + to_pandas OOMs on v5 pr_timeline (6.5M rows) at 12 GB; arrow-backed strings fit
    return {name: pd.read_parquet(hf_hub_download('hao-li/AIDev', f'{name}.parquet', repo_type='dataset',
                                                  revision=REVISION, cache_dir=RAW_PATH),
                                  dtype_backend='pyarrow')
            for name in TABLES}

def task_types(prs):
    # v4+ dropped pr_task_type; BERT labels are cached because labeling needs a GPU (~30 min on T4)
    if TASK_TYPE_PATH.exists():
        tasks = pd.read_parquet(TASK_TYPE_PATH)
        if set(prs['id']) <= set(tasks['id']):
            return tasks
    # child process: torch left in this one plus loaded tables got OOM-killed at 12 GB
    subprocess.run([sys.executable, str(Path(__file__).with_name('task_type_classifier.py')), 'label',
                    '--revision', REVISION, '--out', str(TASK_TYPE_PATH)], check=True)
    return pd.read_parquet(TASK_TYPE_PATH)

def clean_data(tables):
    df = tables['pull_request']

    # nan forced it to be a float, make into nullable integere
    df['repo_id'] = df['repo_id'].astype('Int64')

    # dropping body, title if no value, provides no context
    df['body'] = df['body'].fillna('')
    df['title'] = df['title'].fillna('')

    # labeled before the open filter so the cache covers the whole table
    tasks = task_types(df)

    # timestamps for when PRs were created, closed, and merged
    date_columns = ['created_at', 'closed_at', 'merged_at']

    # open PRs haven't been decided yet, so they aren't failures
    df = df[df['state'] != 'open'].copy()

    for column in date_columns:
        df[column] = pd.to_datetime(df[column])

    # returns a column of boolean values assigned to new column merged
    df['merged'] = df['merged_at'].notna()

    df = df.merge(tasks, on='id', how='left')

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH/'prs.parquet', index=False)
    for name in ['pr_reviews', 'repository', 'pr_timeline']:
        tables[name].to_parquet(PROCESSED_PATH / f'{name}.parquet', index=False)

    return df

if __name__ == '__main__':
    raw_data = load_data()
    clean_data(raw_data)
