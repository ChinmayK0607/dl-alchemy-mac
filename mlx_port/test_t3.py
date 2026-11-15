"""Tests for real-data ordering, knobs, stochastic dropout and evaluation."""
import argparse
import ast
from dataclasses import asdict, replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainConfig, make_config, run_name
from .data import TokenDataset, global_shuffle_prefix_indices, split_path
from .model import Config, Model, depth_model_config, loss
from .run import evaluate
from .schedules import learning_rate
from .train_mlx import Trainer
from .paths import reference_path
from .runtime import configure_test_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/dclm"))
    p.add_argument("--skip-real-data", action="store_true")
    p.add_argument("--verify-source", action="store_true", help="Independently re-fetch 32 original HF rows")
    p.add_argument("--output", type=Path, default=Path("results/T3_tests.json"))
    a = p.parse_args()
    Path(".build/tmp").mkdir(parents=True, exist_ok=True)
    device = configure_test_device(mx)
    selected_device = mx.default_device()
    torch.set_num_threads(4)
    reference = reference_path()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference))
    import data as ref_data
    from utils import create_batches
    from model_config import depth_model_config as ref_depth, LMConfig
    from modeling import AutoregressiveLM, initialize_model
    from optimizers import build_optimizer
    from lr_schedules import build_scheduler
    records = []
    def record(name, **kw):
        r = dict(test=name, status="passed", **kw)
        records.append(r)
        print(json.dumps(r), flush=True)

    # Compare supported unchanged defaults directly to the reference dataclass AST.
    source = ast.parse((reference / "train.py").read_text())
    ref_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "TrainConfig")
    defaults = {n.target.id: ast.literal_eval(n.value) for n in ref_class.body
                if isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Constant)}
    for k, v in asdict(TrainConfig()).items():
        if k in defaults:
            assert v == defaults[k], (k, v, defaults[k])
    assert TrainConfig().data_seed == 42 and TrainConfig().total_steps == 9375 and TrainConfig().eval_steps == 94
    for d in range(4, 10):
        assert asdict(depth_model_config(d)) == ref_depth(d).to_dict()
    record("reference_defaults_and_depths", depths=list(range(4,10)))

    count, max_diff = 0, 0.0
    for schedule in ("cos", "linear", "lin", "constant", "const", "flat", "wsd0", "wsd0.1", "wsd0.5", "wsd1"):
        for total in (1, 7, 103, 320, 9375):
            for warmup_fraction in (0., .01, .2):
                opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=.003)
                scheduler = build_scheduler(opt, schedule, int(total * warmup_fraction), total)
                for step in range(total + 3):
                    actual = learning_rate(step, total, .003, schedule, warmup_fraction)
                    diff = abs(actual - opt.param_groups[0]["lr"])
                    max_diff = max(max_diff, diff)
                    assert diff < 3e-10, (schedule, total, step, diff)
                    opt.step(); scheduler.step()
                    count += 1
    record("schedules_vs_reference", points=count, max_abs_diff=max_diff)

    # Verify the subtle select-THEN-global-shuffle contract against reference
    # materialization using a synthetic full source, multiple seeds and prefixes.
    with tempfile.TemporaryDirectory(dir=".build/tmp", prefix="t3data_") as tmp:
        path = Path(tmp)
        tokens = np.arange(257*16, dtype=np.uint16).reshape(257,16) % 4096
        tokens.tofile(path / "tokens.bin")
        (path / "metadata.json").write_text(json.dumps(dict(num_sequences=257, seq_len=16, dtype="uint16", data_seed=None)))
        os.environ["DL_ALCHEMY_EPHEMERAL_DATA_DIR"] = str(path)
        comparisons = 0
        for seed in (None, 0, 42, 123):
            for n in (23, 128, 257):
                expected = ref_data.PreprocessedTokenDataset(path).select(range(n)).shuffle(seed=seed)
                order = global_shuffle_prefix_indices(257, n, seed)
                assert np.array_equal(order, ref_data.global_shuffle_prefix_indices(257, n, seed)) if seed is not None else True
                actual = tokens[order]
                for i, batch in enumerate(create_batches(expected, 7)):
                    assert np.array_equal(batch.numpy(), actual[i*7:(i+1)*7])
                    comparisons += 1
        del os.environ["DL_ALCHEMY_EPHEMERAL_DATA_DIR"]
    record("global_shuffle_prefix_vs_reference", batch_comparisons=comparisons, seeds=[None,0,42,123])

    # Shared masks isolate dropout placement from backend RNG differences.
    c = Config.tiny()
    torch.manual_seed(42)
    ref = AutoregressiveLM(LMConfig(**asdict(c)), dtype=torch.float32, dropout=.25)
    torch.manual_seed(42); initialize_model(ref)
    port = Model(c, dropout=.25)
    port.load_reference(ref.state_dict())
    ids_np = np.random.default_rng(9).integers(0,128,(2,7),dtype=np.int32)
    ids = mx.array(ids_np)
    masks = []
    def pt_dropout(x, p=0.5, training=True, inplace=False):
        if not training or not p:
            return x
        mask = (np.random.default_rng(100+len(masks)).random(tuple(x.shape)) >= p).astype(np.float32)/(1-p)
        masks.append(mask)
        return x * torch.from_numpy(mask)
    def pt_sdpa(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,**kwargs):
        scores = q @ k.transpose(-1,-2) / math.sqrt(q.shape[-1])
        if is_causal:
            allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, -float("inf"))
        elif attn_mask is not None:
            scores = scores + attn_mask
        return F.dropout(torch.softmax(scores,dim=-1), p=dropout_p, training=bool(dropout_p)) @ v
    with patch.object(F, "dropout", pt_dropout), patch.object(F,"scaled_dot_product_attention",pt_sdpa):
        expected = ref(torch.from_numpy(ids_np).long())
        expected.mean().backward()
    assert len(masks) == 3*c.num_hidden_layers
    for layer in range(c.num_hidden_layers):
        assert masks[3*layer].shape == (2,c.num_attention_heads,7,7)
        assert masks[3*layer+1].shape == masks[3*layer+2].shape == (2,7,c.hidden_size)
    used = []
    def fixed_dropout(module, x):
        if not module.training or module._p_1 == 1:
            return x
        mask = masks[len(used)]
        used.append(x.shape)
        assert x.shape == mask.shape
        return x * mx.array(mask)
    with patch.object(nn.Dropout,"__call__",fixed_dropout):
        actual = np.array(port(ids))
        used.clear()
        def objective(params):
            from mlx.utils import tree_unflatten
            port.update(tree_unflatten(list(params.items())))
            return port(ids).mean()
        _, grads = mx.value_and_grad(objective)(dict(tree_flatten(port.parameters())))
        grad_diff = max(float(np.max(np.abs(p.grad.numpy()-np.array(grads[n])))) for n,p in ref.named_parameters())
    diff = float(np.max(np.abs(expected.detach().numpy()-actual)))
    assert diff < 2e-4 and grad_diff < 2e-4, (diff,grad_diff)
    ref.eval(); port.eval()
    with torch.no_grad():
        eval_diff = float(np.max(np.abs(ref(torch.from_numpy(ids_np).long()).numpy()-np.array(port(ids)))))
    assert eval_diff < 2e-4
    record("dropout_placement_shared_masks", sites=len(masks), logits_max_abs_diff=diff,
           grad_max_abs_diff=grad_diff, eval_logits_max_abs_diff=eval_diff)

    def dropout_curve():
        m = Model(c, dropout=.2)
        m.initialize(77)
        t = Trainer(m,1,2,compiled=True,total_steps=20)
        t.materialize()
        values, grads = t.grad(t.params,ids)
        first = float(values.item())
        second = float(t.grad(t.params,ids)[0].item())
        assert abs(first-second) > 1e-5  # compiled RNG must advance
        ev1 = float(t.evaluate(ids).item()); ev2 = float(t.evaluate(ids).item())
        assert ev1 == ev2  # eval disables all dropout, including after a traced train
        curve = []
        for _ in range(3):
            v,n = t.step(ids,lr=.001)
            curve.append(float(v.item()))
            assert math.isfinite(float(n.item()))
            assert float(t.evaluate(ids).item()) == float(t.evaluate(ids).item())
        return first,second,curve
    stochastic1,stochastic2 = dropout_curve(),dropout_curve()
    assert stochastic1 == stochastic2
    record("compiled_dropout_rng_and_eval", first_losses=stochastic1[:2], curve=stochastic1[2], reproducible=True)

    # Non-default optimizer settings + a final partial micro-batch match reference.
    torch.manual_seed(42)
    ref = AutoregressiveLM(LMConfig(**asdict(c)), dtype=torch.float32)
    torch.manual_seed(42); initialize_model(ref)
    port = Model(c); port.load_reference(ref.state_dict())
    opt = build_optimizer(ref,"adamw",.002,.23,.8,.97)
    trainer = Trainer(port,2,4,compiled=True,total_steps=7,learning_rate=.002,lr_schedule="cos",
                      warmup_percent=.2,weight_decay=.23,beta1=.8,beta2=.97,grad_norm=.3)
    scheduler = build_scheduler(opt,"cos",1,7)
    data_np = np.random.default_rng(1).integers(0,128,(3,7),dtype=np.int32)
    max_loss_diff = 0.
    for step in range(7):
        opt.zero_grad()
        expected = 0.
        for batch in (data_np[:2],data_np[2:]):
            logits = ref(torch.from_numpy(batch).long())[:,:-1].float()
            value = F.cross_entropy(logits.reshape(-1,128),torch.from_numpy(batch[:,1:].copy()).long().reshape(-1))/2
            value.backward(); expected += value.item()
        torch.nn.utils.clip_grad_norm_(ref.parameters(),.3)
        opt.step();scheduler.step()
        v,n = trainer.step_micro_batches([mx.array(data_np[:2]),mx.array(data_np[2:])])
        max_loss_diff=max(max_loss_diff,abs(expected-float(v.item())))
    assert max_loss_diff < 2e-4
    weight_diff = max(float(np.max(np.abs(p.detach().numpy()-np.array(trainer.params[n])))) for n,p in ref.named_parameters())
    assert weight_diff < 2e-4
    record("custom_optimizer_partial_accumulation", steps=7, max_loss_diff=max_loss_diff, weight_max_abs_diff=weight_diff)

    # Evaluation weights NLL by number of next tokens, not batches.
    class FakeTrainer:
        def evaluate(self, batch):
            return mx.array(float(np.array(batch).mean()))
    data_np = np.arange(35,dtype=np.int32).reshape(5,7)
    actual = evaluate(FakeTrainer(),data_np,2,context=7)
    assert abs(actual-data_np.mean()) < 1e-6
    record("eval_partial_batch_weighting", rows=5, micro_batch=2, result=actual)
    assert make_config("local").total_steps==320 and make_config("smoke").total_steps==16
    assert make_config("local").eval_steps==81 and make_config("smoke").eval_steps==17
    assert len({run_name(replace(make_config("local"),**v)) for v in ({},{"dropout":.1},{"lr_schedule":"cos"},{"model_seed":1})})==4
    record("preset_intervals_and_run_names", local_steps=320,local_interval=81,smoke_steps=16)

    # Real runner integration with an explicitly synthetic fixture: artifacts,
    # partial validation, checkpoint reload and no-overwrite behavior.
    from .run import train as run_train
    from .data import load_datasets
    with tempfile.TemporaryDirectory(dir=".build/tmp", prefix="t3runner_") as tmp:
        fixture=Path(tmp)/"data"
        for split, rows in (("train",20480),("val",1000)):
            directory=split_path(fixture,split,42)
            directory.mkdir(parents=True)
            tokens=np.memmap(directory/"tokens.bin",mode="w+",dtype=np.uint16,shape=(rows,1024))
            tokens[:]=np.arange(1024,dtype=np.uint16)%128
            tokens.flush()
            (directory/"metadata.json").write_text(json.dumps(dict(num_sequences=rows,seq_len=1024,
                dtype="uint16",data_seed=42 if split=="train" else None,synthetic_test_fixture=True)))
        summary=run_train(make_config("local"),device="cpu",data_dir=fixture,output_dir=Path(tmp)/"runs",
                          max_steps=2,verification_context=16,verification_val_sequences=9,
                          wandb="disabled",save_weights=True)
        run_dir=Path(summary["run_dir"])
        weights=mx.load(str(run_dir/"final.safetensors"))
        assert all(v.dtype==mx.float32 for v in weights.values())
        assert sum(v.size for v in weights.values())==5901056
        mx.set_default_device(selected_device)
        restored=Model(depth_model_config(4),precision="mp")
        restored.load_weights(list(weights.items()),strict=True)
        check_trainer=Trainer(restored,8,64,compiled=True)
        _,val=load_datasets(fixture,20480,42)
        restored_eval=evaluate(check_trainer,val,8,context=16,limit=9)
        assert abs(restored_eval-summary["final_eval_loss"])<1e-5
        metrics=[json.loads(line) for line in (run_dir/"metrics.jsonl").read_text().splitlines()]
        assert len(metrics)==2 and metrics[-1]["tokens"]==2*64*16 and "eval_loss" in metrics[-1]
        try:
            run_train(make_config("local"),device="cpu",data_dir=fixture,output_dir=Path(tmp)/"runs",
                      max_steps=2,verification_context=16,verification_val_sequences=9,
                          wandb="disabled",save_weights=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing run was overwritten")
        record("runner_artifacts_and_checkpoint_reload",parameters=5901056,
               eval_loss_abs_diff=abs(restored_eval-summary["final_eval_loss"]),
               steps=2,fixture="synthetic",overwrite_protected=True)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not a.skip_real_data:
        checks=0
        for n in (17,1024,20480,100000):
            actual = TokenDataset(split_path(a.data_dir,"train",42),n,42)
            ref = ref_data.PreprocessedTokenDataset(actual.path).select(range(n)).shuffle(seed=42)
            for i,batch in enumerate(create_batches(ref,7)):
                if i>=8:
                    break
                assert np.array_equal(batch.numpy(),actual[i*7:(i+1)*7])
                checks+=1
        for split in ("train","val"):
            path=split_path(a.data_dir,split,42)
            actual=TokenDataset(path)
            ref=ref_data.PreprocessedTokenDataset(path)
            for i,batch in enumerate(create_batches(ref,8)):
                if i>=8:
                    break
                assert np.array_equal(batch.numpy(),actual[i*8:(i+1)*8])
                checks+=1
        source_n=actual.metadata.get("source_num_sequences",9600000)
        indices=np.load(split_path(a.data_dir,"train",42)/"source_rows.npy")
        assert np.array_equal(indices,ref_data.global_shuffle_prefix_indices(9600000,len(indices),42))
        record("real_dclm_first_batches_token_exact",batch_comparisons=checks,prefixes=[17,1024,20480,100000],val_sequences=1000)
        if a.verify_source:
            from concurrent.futures import ThreadPoolExecutor
            import urllib.request
            from .data import REPO, REVISION
            original_rows=ref_data.global_shuffle_prefix_indices(9600000,32,42)
            cached=TokenDataset(split_path(a.data_dir,"train",42),32,42)
            def check_source(item):
                i,row_id=item
                first,last=int(row_id)*2048,(int(row_id)+1)*2048-1
                url=f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/train/tokens.bin?verify_row={row_id}"
                with urllib.request.urlopen(urllib.request.Request(url,headers={"Range":f"bytes={first}-{last}"}),timeout=30) as r:
                    assert r.status==206 and r.headers["Content-Range"]==f"bytes {first}-{last}/19660800000"
                    payload=r.read(2049)
                assert len(payload)==2048
                assert np.array_equal(np.frombuffer(payload,dtype=np.uint16),cached[i])
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(check_source,enumerate(original_rows)))
            record("independent_remote_source_rows",rows=32,downloaded_token_bytes=32*2048)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status="passed",device=device,real_data_checked=not a.skip_real_data,records=records),indent=2)+"\n")


if __name__ == "__main__":
    main()
