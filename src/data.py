from datasets import load_dataset
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / 'data' / 'raw'
PROCESSED_PATH = ROOT / 'data' / 'processed'

def load_data():
    ds = load_dataset("hao-li/AIDev", "all_pull_request", cache_dir=RAW_PATH) # cache dir saves to a directory
    return ds

def clean_data(ds):
    df = ds['train'].select(range(100_000)).to_pandas()

    # nan forced it to be a float, make into nullable integere
    df['repo_id'] = df['repo_id'].astype('Int64')

    # dropping body, title if no value, provides no context
    df = df.dropna(subset=['body','title'])

    # timestamps for when PRs were created, closed, and merged
    date_columns = ['created_at', 'closed_at', 'merged_at']

    for column in date_columns:
        df[column] = pd.to_datetime(df[column])

    # returns a column of boolean values assigned to new column merged 
    df['merged'] = df['merged_at'].notna()

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH/'prs.parquet', index=False)

    return df

if __name__ == '__main__':
    raw_data = load_data()
    clean_data(raw_data)