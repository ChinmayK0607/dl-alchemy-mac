"""Bounded DCLM downloads and the reference's global-shuffle prefix contract.

Only requested rows are fetched with verified HTTP byte ranges. The cache is
already globally shuffled: never shuffle its small prefix a second time.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

REPO = "kothasuhas/dl_alchemy_seq9p6m_context1024"
REVISION = "8d066a12a614ce8641c661874960bf8567bf0f19"
DEFAULT_ROOT = Path("data/dclm")


def global_shuffle_prefix_indices(source_num_sequences, output_num_sequences, seed):
    if not 0 < output_num_sequences <= source_num_sequences:
        raise ValueError("Invalid sequence prefix length")
    if seed is None:
        return np.arange(output_num_sequences, dtype=np.int64)
    return np.random.default_rng(int(seed)).permutation(source_num_sequences)[:output_num_sequences]


def split_path(root, split, seed=42):
    return Path(root) / (f"train_ds{seed}" if split == "train" else "val")


class TokenDataset:
    def __init__(self, path, num_sequences=None, seed=None):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        n, s = int(self.metadata["num_sequences"]), int(self.metadata["seq_len"])
        if seed is not None and self.metadata.get("data_seed") != seed:
            raise ValueError("Cache seed mismatch; download the requested global prefix")
        self.num_sequences = n if num_sequences is None else num_sequences
        if not 0 < self.num_sequences <= n:
            raise ValueError(f"Requested {self.num_sequences} rows, cache has {n}; run mlx_port.data")
        self.tokens = np.memmap(self.path / "tokens.bin", mode="r", dtype=self.metadata["dtype"], shape=(n, s))
        if self.tokens.min() < 0 or self.tokens.max() >= 4096:
            raise ValueError("Token IDs outside vocabulary")
        self.seq_len = s

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.tokens[:self.num_sequences][idx]

    def batches(self, batch_size):
        for start in range(0, len(self), batch_size):
            yield np.asarray(self[start:start+batch_size], dtype=np.int32)


def load_datasets(root=DEFAULT_ROOT, num_train_sequences=600000, data_seed=42):
    train = TokenDataset(split_path(root, "train", data_seed), num_train_sequences, data_seed)
    val = TokenDataset(split_path(root, "val"), 1000)
    if train.seq_len != 1024 or val.seq_len != 1024 or len(val) != 1000:
        raise ValueError("DCLM requires context 1024 and 1000 validation sequences")
    return train, val


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(root=DEFAULT_ROOT, num_sequences=100000, seed=42, workers=32, merge_gap=8):
    """Resume a seed-specific prefix, <= 2 GB network payload and local storage.

    A full base dataset is 19.66 GB. Random access here preserves its ordering
    without materializing that file. Per-range Content-Range/length is mandatory;
    the validation file additionally gets the publisher's full SHA256 check.
    """
    if not 1 <= workers <= 128 or merge_gap < 0:
        raise ValueError("workers must be in [1,128], merge_gap must be nonnegative")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
    def small_file(name):
        with urllib.request.urlopen(base + name, timeout=60) as r:
            return r.read(1 << 20)
    manifest_raw = small_file("manifest.json")
    (root / "source_manifest.json").write_bytes(manifest_raw)
    manifest = json.loads(manifest_raw)
    files = {x["path"]: x for x in manifest["files"]}
    metadata = {}
    for split in ("train", "val"):
        raw = small_file(f"{split}/metadata.json")
        if hashlib.sha256(raw).hexdigest() != files[f"{split}/metadata.json"]["sha256"]:
            raise ValueError("Source metadata digest mismatch")
        (root / f"source_{split}_metadata.json").write_bytes(raw)
        metadata[split] = json.loads(raw)
    val_path = split_path(root, "val")
    val_path.mkdir(exist_ok=True)
    vf = val_path / "tokens.bin"
    info = files["val/tokens.bin"]
    if not vf.exists() or _sha(vf) != info["sha256"]:
        with urllib.request.urlopen(base + "val/tokens.bin", timeout=60) as r:
            payload = r.read(info["size_bytes"] + 1)
        if len(payload) != info["size_bytes"] or hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise ValueError("Validation digest mismatch")
        vf.write_bytes(payload)
    (val_path / "metadata.json").write_text(json.dumps(metadata["val"], indent=2) + "\n")
    dest = split_path(root, "train", seed)
    dest.mkdir(exist_ok=True)
    ready = dest / "metadata.json"
    if ready.exists():
        old = json.loads(ready.read_text())
        if old["num_sequences"] >= num_sequences and old["source_revision"] == REVISION:
            print(f"Already complete: {dest}", flush=True)
            return old
        raise ValueError("Existing completed cache differs; choose another --data-dir")
    source_n = metadata["train"]["num_sequences"]
    row_bytes = metadata["train"]["seq_len"] * np.dtype(metadata["train"]["dtype"]).itemsize
    indices = global_shuffle_prefix_indices(source_n, num_sequences, seed)
    index_file = dest / "source_rows.npy"
    if index_file.exists():
        if not np.array_equal(np.load(index_file), indices):
            raise ValueError("In-progress cache differs; choose another --data-dir")
    else:
        np.save(index_file, indices)
    order = np.argsort(indices)
    groups = []
    start = 0
    for i in range(1, num_sequences + 1):
        if i == num_sequences or indices[order[i]] - indices[order[i-1]] > merge_gap + 1:
            groups.append(order[start:i])
            start = i
    payload_bytes = sum((int(indices[g[-1]]) - int(indices[g[0]]) + 1) * row_bytes for g in groups)
    if payload_bytes + info["size_bytes"] > 1_900_000_000:
        raise ValueError("Download exceeds 1.9 GB payload cap; reduce rows/merge-gap")
    token_file = dest / "tokens.bin"
    tokens = np.memmap(token_file, mode="r+" if token_file.exists() else "w+",
                       dtype=metadata["train"]["dtype"], shape=(num_sequences, 1024))
    done_path = dest / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(len(groups), dtype=bool)
    plan_path = dest / "download_plan.json"
    plan = dict(num_sequences=num_sequences, seed=seed, merge_gap=merge_gap,
                groups=len(groups), payload_bytes=payload_bytes, revision=REVISION)
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume download plan mismatch")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    # Resolve once, then use persistent CDN connections rather than 100k Hub calls.
    with urllib.request.urlopen(urllib.request.Request(base + "train/tokens.bin?download=true", method="HEAD"), timeout=60) as r:
        signed = urllib.parse.urlsplit(r.url)
    local = threading.local()
    def fetch(job):
        j, g = job
        lo, hi = int(indices[g[0]]), int(indices[g[-1]])
        first, last = lo * row_bytes, (hi + 1) * row_bytes - 1
        for attempt in range(6):
            try:
                if not hasattr(local, "conn"):
                    local.conn = http.client.HTTPSConnection(signed.netloc, timeout=30)
                local.conn.request("GET", signed.path + "?" + signed.query,
                                   headers={"Range": f"bytes={first}-{last}"})
                response = local.conn.getresponse()
                expected = f"bytes {first}-{last}/{files['train/tokens.bin']['size_bytes']}"
                if response.status != 206 or response.getheader("Content-Range") != expected:
                    raise ValueError(f"Invalid range response: {response.status}; expected {expected}")
                raw = response.read(last - first + 2)
                if len(raw) != last - first + 1:
                    raise ValueError("Truncated range")
                rows = np.frombuffer(raw, dtype=metadata["train"]["dtype"]).reshape(-1, 1024)
                tokens[g] = rows[indices[g] - lo]
                return j
            except Exception:
                if hasattr(local, "conn"):
                    local.conn.close()
                    del local.conn
                if attempt == 5:
                    raise
                time.sleep(min(8, 0.5 * 2**attempt))
    jobs = [(j, g) for j, g in enumerate(groups) if not done[j]]
    print(json.dumps(dict(download_plan=plan, remaining_ranges=len(jobs))), flush=True)
    started = time.perf_counter()
    def save_done():
        temporary = done_path.with_name("done.tmp.npy")
        np.save(temporary, done)
        temporary.replace(done_path)
    # map is ordered and bounded by executor work queue; progress checkpoints make
    # an interrupted download resumable without re-fetching completed prefixes.
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for count, j in enumerate(pool.map(fetch, jobs), 1):
                done[j] = True
                if count % 1000 == 0:
                    tokens.flush()
                    save_done()
                    print(f"ranges {int(done.sum())}/{len(groups)}; {time.perf_counter()-started:.1f}s", flush=True)
    finally:
        tokens.flush()
        save_done()
    if not done.all():
        raise RuntimeError("Incomplete download")
    result = dict(metadata["train"], num_sequences=num_sequences, data_seed=seed,
                  source_num_sequences=source_n, source_revision=REVISION, source_repo=REPO,
                  shuffle="global_row_permutation_prefix", download_payload_bytes=payload_bytes,
                  tokens_sha256=_sha(token_file), source_rows_sha256=_sha(index_file),
                  numpy_version=np.__version__)
    temporary = ready.with_name("metadata.json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(ready)
    print(json.dumps(dict(status="ready", path=str(dest), **result)), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--num-train-sequences", type=int, default=100000)
    p.add_argument("--data-seed", type=lambda s: None if s.lower() == "none" else int(s), default=42)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--merge-gap", type=int, default=8)
    a = p.parse_args()
    download(a.data_dir, a.num_train_sequences, a.data_seed, a.workers, a.merge_gap)


if __name__ == "__main__":
    main()
