"""Measure the held-out training-unigram baseline, in the reference's NLL units."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_ROOT, load_datasets


def histogram(dataset, chunk_rows=2048):
    counts = np.zeros(4096, dtype=np.int64)
    for batch in dataset.batches(chunk_rows):
        # Only targets seen by the causal objective, never position zero.
        counts += np.bincount(batch[:,1:].reshape(-1), minlength=4096)
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--num-train-sequences",type=int,default=20480)
    p.add_argument("--data-seed",type=int,default=42)
    p.add_argument("--output",type=Path,default=Path("results/T3_unigram.json"))
    a=p.parse_args()
    train,val=load_datasets(a.data_dir,a.num_train_sequences,a.data_seed)
    train_counts,val_counts=histogram(train),histogram(val)
    prob=(train_counts+1)/(train_counts.sum()+4096)  # Laplace smoothing
    val_prob=val_counts/val_counts.sum()
    positive=val_prob>0
    result=dict(num_train_sequences=len(train),num_val_sequences=len(val),
                train_target_tokens=int(train_counts.sum()),val_target_tokens=int(val_counts.sum()),
                train_unigram_val_nll_nats=float(-(val_counts*np.log(prob)).sum()/val_counts.sum()),
                empirical_val_unigram_entropy_nats=float(-(val_prob[positive]*np.log(val_prob[positive])).sum()),
                uniform_nll_nats=float(np.log(4096)),smoothing="add one per vocabulary ID")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
