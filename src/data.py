from datasets import load_dataset
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / 'data' / 'raw'
PROCESSED_PATH = ROOT / 'data' / 'processed'

TABLES = ['pull_request', 'pr_task_type', 'pr_reviews', 'repository', 'pr_timeline']

def load_data():
    return {name: load_dataset("hao-li/AIDev", name, revision="v3", cache_dir=RAW_PATH)['train'].to_pandas()
            for name in TABLES}

def clean_data(tables):
    df = tables['pull_request']

    # nan forced it to be a float, make into nullable integere
    df['repo_id'] = df['repo_id'].astype('Int64')

    # dropping body, title if no value, provides no context
    df['body'] = df['body'].fillna('')
    df['title'] = df['title'].fillna('')

    # timestamps for when PRs were created, closed, and merged
    date_columns = ['created_at', 'closed_at', 'merged_at']

    # open PRs haven't been decided yet, so they aren't failures
    df = df[df['state'] != 'open'].copy()

    for column in date_columns:
        df[column] = pd.to_datetime(df[column])

    # returns a column of boolean values assigned to new column merged 
    df['merged'] = df['merged_at'].notna()

    tasks = tables['pr_task_type'][['id', 'type']].rename(columns={'type': 'task_type'})
    df = df.merge(tasks, on='id', how='left')

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH/'prs.parquet', index=False)
    for name in ['pr_reviews', 'repository', 'pr_timeline']:
        tables[name].to_parquet(PROCESSED_PATH / f'{name}.parquet', index=False)

    return df

if __name__ == '__main__':
    raw_data = load_data()
    clean_data(raw_data)