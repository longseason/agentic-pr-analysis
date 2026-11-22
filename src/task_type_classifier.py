"""BERT task-type classifier: trains on AIDev v3 labels, labels newer AIDev releases.

Usage:
    python task_type_classifier.py train [--out DIR] [--push]
    python task_type_classifier.py label [--revision v5] [--table pull_request] [--limit N] [--out FILE]

Labeling streams one parquet row group (200K PRs) at a time and saves each as a chunk, so a killed
run resumes where it stopped when rerun with the same --out.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF = 'hf://datasets/hao-li/AIDev'
BASE_MODEL = 'bert-base-uncased'
MODEL_ID = 'Hydragon123/aidev-task-type-bert'
MAX_LEN = 256
EPOCHS = 3
BATCH = 32
LR = 2e-5
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent.parent


def text_of(df):
    return (df['title'].fillna('') + '\n\n' + df['body'].fillna('')).tolist()


def batches(tok, texts, labels=None, bs=BATCH, shuffle=False):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        enc = tok([texts[j] for j in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
        if labels is not None:
            enc['labels'] = torch.tensor(labels[b])
        yield b, enc


@torch.no_grad()
def predict(model, tok, texts, bs=256):
    """predict returns softmax probabilities, one row per text."""
    model.eval()
    # length-sorted batches cut padding waste
    order = np.argsort([len(t) for t in texts])
    sorted_texts = [texts[i] for i in order]
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    t0 = time.time()
    for k, (b, enc) in enumerate(batches(tok, sorted_texts, bs=bs)):
        enc = {key: v.to(DEVICE) for key, v in enc.items()}
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == 'cuda'):
            logits = model(**enc).logits
        probs[order[b]] = torch.softmax(logits.float(), -1).cpu().numpy()
        if k % 100 == 0:
            print(f'predict {b[-1] + 1}/{len(texts)} {time.time() - t0:.0f}s', flush=True)
    return probs


def load_labeler(model_id=MODEL_ID):
    """load_labeler returns (model, tokenizer, v3 label series indexed by PR id)."""
    if DEVICE == 'cpu':
        print('WARNING: no GPU, labeling will be slow (~2.6 PRs/s)', flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE)
    known = pd.read_parquet(f'{HF}@v3/pr_task_type.parquet', columns=['id', 'type']).set_index('id')['type']
    return model, tok, known


def label_prs(prs, labeler):
    """label_prs assigns a task type to every row of a frame with id, title, body."""
    model, tok, known = labeler
    classes = np.array([model.config.id2label[i] for i in range(model.config.num_labels)])
    out = pd.DataFrame({'id': prs['id'].to_numpy(), 'task_type': None,
                        'task_type_confidence': np.float32(1.0), 'task_type_source': 'v3_label'})
    # PRs carrying a v3 LLM label keep it; model only fills the gap
    has = out['id'].isin(known.index).to_numpy()
    out.loc[has, 'task_type'] = out.loc[has, 'id'].map(known).to_numpy()
    if (~has).any():
        probs = predict(model, tok, text_of(prs[~has]))
        out.loc[~has, 'task_type'] = classes[probs.argmax(1)]
        out.loc[~has, 'task_type_confidence'] = probs.max(1)
        out.loc[~has, 'task_type_source'] = 'bert'
    return out


def label_table(source, out_path, model_id=MODEL_ID, limit=None):
    """label_table labels a pull_request parquet one row group at a time, resuming from saved chunks."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_path = Path(out_path)
    chunk_dir = out_path.with_name(out_path.stem + '_chunks')
    chunk_dir.mkdir(parents=True, exist_ok=True)
    fh = HfFileSystem().open(source.removeprefix('hf://')) if source.startswith('hf://') else open(source, 'rb')
    pf = pq.ParquetFile(fh)
    labeler, done, t0 = None, 0, time.time()
    for i in range(pf.num_row_groups):
        if limit is not None and done >= limit:
            break
        chunk = chunk_dir / f'{i:04d}.parquet'
        n = pf.metadata.row_group(i).num_rows
        if chunk.exists():
            print(f'chunk {i + 1}/{pf.num_row_groups} cached', flush=True)
            done += n
            continue
        labeler = labeler or load_labeler(model_id)
        prs = pf.read_row_group(i, columns=['id', 'title', 'body']).to_pandas()
        if limit is not None:
            prs = prs.head(limit - done)
        # write-then-rename so a killed run never leaves a half chunk that looks complete
        tmp = chunk.with_suffix('.tmp')
        label_prs(prs, labeler).to_parquet(tmp, index=False)
        tmp.rename(chunk)
        done += len(prs)
        print(f'chunk {i + 1}/{pf.num_row_groups} labeled, {done} rows, {time.time() - t0:.0f}s', flush=True)
    out = pd.concat([pd.read_parquet(c) for c in sorted(chunk_dir.glob('*.parquet'))], ignore_index=True)
    out.to_parquet(out_path, index=False)
    for c in chunk_dir.glob('*'):
        c.unlink()
    chunk_dir.rmdir()
    return out


def train(out_dir, push=False):
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from transformers import get_linear_schedule_with_warmup

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    labels = pd.read_parquet(f'{HF}@v3/pr_task_type.parquet', columns=['id', 'type'])
    v3 = pd.read_parquet(f'{HF}@v3/pull_request.parquet', columns=['id', 'title', 'body'])
    train_df = v3.merge(labels, on='id', how='inner')
    classes = sorted(train_df['type'].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = train_df['type'].map(c2i).to_numpy()
    print('train rows', len(train_df), train_df['type'].value_counts().to_dict(), flush=True)

    tr, va = train_test_split(np.arange(len(train_df)), test_size=0.1, stratify=y, random_state=SEED)
    texts = text_of(train_df)
    tr_texts, va_texts = [texts[i] for i in tr], [texts[i] for i in va]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=c2i).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(tr) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=DEVICE == 'cuda')

    best_f1, history, report = -1.0, [], None
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for k, (_, enc) in enumerate(batches(tok, tr_texts, y[tr], shuffle=True)):
            enc = {key: v.to(DEVICE) for key, v in enc.items()}
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == 'cuda'):
                loss = model(**enc).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if k % 200 == 0:
                print(f'ep {ep} step {k} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
        pred = predict(model, tok, va_texts).argmax(1)
        mf1 = f1_score(y[va], pred, average='macro')
        history.append({'epoch': ep, 'acc': float((pred == y[va]).mean()), 'macro_f1': mf1,
                        'weighted_f1': f1_score(y[va], pred, average='weighted')})
        print(history[-1], flush=True)
        if mf1 > best_f1:
            best_f1 = mf1
            model.save_pretrained(out_dir / 'model')
            tok.save_pretrained(out_dir / 'model')
            report = classification_report(y[va], pred, labels=range(len(classes)), target_names=classes,
                                           zero_division=0, output_dict=True)

    json.dump({'history': history, 'best_val_report': report, 'n_train': len(tr), 'n_val': len(va),
               'model': BASE_MODEL, 'max_len': MAX_LEN}, open(out_dir / 'metrics.json', 'w'), indent=2)
    if push:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=out_dir / 'model', repo_id=MODEL_ID, commit_message='retrain')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--out', type=Path, default=ROOT / 'data' / 'models' / 'task_type')
    t.add_argument('--push', action='store_true', help=f'upload best checkpoint to {MODEL_ID}')
    lab = sub.add_parser('label')
    lab.add_argument('--revision', default='v5')
    lab.add_argument('--table', default='pull_request', help='pull_request or all_pull_request')
    lab.add_argument('--limit', type=int)
    lab.add_argument('--model', default=MODEL_ID)
    lab.add_argument('--out', type=Path)
    lab.add_argument('--source', help='parquet path or hf:// URI; overrides --revision/--table')
    args = ap.parse_args()

    if args.cmd == 'train':
        train(args.out, args.push)
        return
    source = args.source or f'{HF}@{args.revision}/{args.table}.parquet'
    path = args.out or ROOT / 'data' / 'processed' / f'{args.table}_task_type_{args.revision}.parquet'
    out = label_table(source, path, args.model, args.limit)
    print('wrote', path, len(out), out['task_type_source'].value_counts().to_dict())


if __name__ == '__main__':
    main()
