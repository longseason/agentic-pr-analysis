from datasets import load_dataset

def load_data():
    ds = load_dataset("hao-li/AIDev", "all_pull_request")
    return ds

def clean_data(ds):
    return ds # for now, we'll keep it like this