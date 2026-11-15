"""Parity: monkeypatch reference instances only; never change its files."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import MethodType

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import torch
import torch.nn.functional as F

from .model import Config, Model, VARIANTS, loss
from .paths import reference_path
from .train_mlx import Trainer, decay_parameter
from .runtime import configure_test_device


def patch_reference(ref, variant):
    """Patch the lecture equations, preserving every original parameter/module."""
    if variant == "no_prenorm":
        for layer in ref.model.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm.forward = MethodType(lambda self,x:x,norm)
    if variant == "no_rope":
        def identity_rotations(self, hidden, positions):
            shape = (*positions.shape, self.head_dim)
            return torch.ones(shape,dtype=hidden.dtype), torch.zeros(shape,dtype=hidden.dtype)
        ref.model.rotary_emb.forward = MethodType(identity_rotations,ref.model.rotary_emb)

    if variant in ("no_residuals", "no_mlp", "no_attention"):
        def block(self, hidden_states, position_embeddings, attention_mask=None):
            x = hidden_states
            if variant == "no_attention":
                a = torch.zeros_like(x)
            else:
                a = self.self_attn(self.input_layernorm(x),position_embeddings,attention_mask)
                a = F.dropout(a,p=self.dropout,training=self.training)
            h = a if variant == "no_residuals" else x+a
            m = torch.zeros_like(h) if variant == "no_mlp" else F.dropout(
                self.mlp(self.post_attention_layernorm(h)),p=self.dropout,training=self.training)
            return m if variant == "no_residuals" else h+m
        for layer in ref.model.layers:
            layer.forward = MethodType(block,layer)
    if variant == "mixed_residual":
        def backbone(self,input_ids,attention_mask=None,position_ids=None):
            x = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1])[None].expand(input_ids.shape[0],-1)
            positions = self.rotary_emb(x,position_ids)
            previous = None
            for layer in self.layers:
                old = x
                if previous is None:
                    x = layer(x,positions,attention_mask)
                else:
                    a = F.dropout(layer.self_attn(layer.input_layernorm(x),positions,attention_mask),
                                  p=layer.dropout,training=layer.training)
                    m = F.dropout(layer.mlp(layer.post_attention_layernorm(x+a)),
                                  p=layer.dropout,training=layer.training)
                    x = .5*x + .5*previous + a + m
                previous = old
            return self.norm(x)
        ref.model.forward = MethodType(backbone,ref.model)
    return ref


def ce(ref, tokens):
    return F.cross_entropy(ref(tokens)[:,:-1].reshape(-1,ref.config.vocab_size),tokens[:,1:].reshape(-1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("results/T4_variants.json"))
    args=p.parse_args()
    device = configure_test_device(mx)
    torch.set_num_threads(4)
    reference = reference_path()
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(reference))
    from model_config import LMConfig
    from modeling import AutoregressiveLM, initialize_model
    from optimizers import build_optimizer
    c=replace(Config.tiny(),num_hidden_layers=3)
    tokens=np.random.default_rng(19).integers(0,c.vocab_size,(2,9),dtype=np.int32)
    pt=torch.from_numpy(tokens).long()
    ids=mx.array(tokens)
    records=[]
    for variant in VARIANTS:
        torch.manual_seed(42)
        ref=AutoregressiveLM(LMConfig(**asdict(c)),dtype=torch.float32)
        torch.manual_seed(42); initialize_model(ref)
        initial={n:p.detach().numpy().copy() for n,p in ref.named_parameters()}
        patch_reference(ref,variant)
        model=Model(c,variant=variant)
        model.load_reference(ref.state_dict())
        assert {n:tuple(p.shape) for n,p in ref.named_parameters()} == {n:tuple(p.shape) for n,p in tree_flatten(model.parameters())}
        expected=ref(pt)
        actual=np.array(model(ids))
        logits_diff=float(np.max(np.abs(expected.detach().numpy()-actual)))
        np.testing.assert_allclose(actual,expected.detach().numpy(),atol=3e-5,rtol=3e-5)
        initial_loss=ce(ref,pt)
        initial_loss.backward()
        trainer=Trainer(model,1,2,compiled=True,total_steps=10,chunk_size=7,
                        learning_rate=.0003,warmup_percent=0)
        trainer.materialize()
        value,grads=trainer.grad(trainer.params,ids)
        loss_diff=abs(float(value.item())-initial_loss.item())
        assert loss_diff<2e-5
        grad_diff=0.
        unused=[]
        for n,param in ref.named_parameters():
            g=np.array(grads[n])
            assert (param.grad is None)==model.unused_parameter(n),n
            if param.grad is None:
                unused.append(n)
                assert np.count_nonzero(g)==0,n
            else:
                np.testing.assert_allclose(g,param.grad.numpy(),atol=4e-5,rtol=4e-4,err_msg=n)
                grad_diff=max(grad_diff,float(np.max(np.abs(g-param.grad.numpy()))))
        model.update(tree_unflatten(list(trainer.params.items())))
        mask=np.ones_like(tokens);mask[0,-2:]=0
        positions=np.broadcast_to(np.arange(9,dtype=np.int32)+7,tokens.shape).copy()
        with torch.no_grad():
            masked=ref(pt,attention_mask=torch.from_numpy(mask),position_ids=torch.from_numpy(positions)).numpy()
        masked_diff=float(np.max(np.abs(masked-np.array(model(ids,mx.array(mask),mx.array(positions))))))
        assert masked_diff<1e-4
        opt=build_optimizer(ref,"adamw",.0003,.1,.9,.95)
        curve=[]
        for step in range(10):
            opt.zero_grad(set_to_none=True)
            v=ce(ref,pt);v.backward()
            norm=torch.nn.utils.clip_grad_norm_(ref.parameters(),1.)
            lr=.0003*(1-step/10)
            for group in opt.param_groups:
                group['lr']=lr
            opt.step()
            mv,mn=trainer.step(ids,lr=lr)
            curve.append(dict(step=step+1,torch_loss=v.item(),mlx_loss=float(mv.item()),
                              torch_grad_norm=norm.item(),mlx_grad_norm=float(mn.item())))
        curve_diff=max(abs(r['torch_loss']-r['mlx_loss']) for r in curve)
        assert curve_diff<2e-4,(variant,curve_diff)
        weights_diff=max(float(np.max(np.abs(p.detach().numpy()-np.array(trainer.params[n])))) for n,p in ref.named_parameters())
        assert weights_diff<3e-4,(variant,weights_diff)
        for n in unused:
            np.testing.assert_array_equal(initial[n],np.array(trainer.params[n]))
            np.testing.assert_array_equal(initial[n],ref.state_dict()[n].numpy())
            assert np.count_nonzero(np.array(trainer.m[n]))==0 and np.count_nonzero(np.array(trainer.v[n]))==0
        # Explicit PyTorch semantics proof: None skips decay, explicit zero decays.
        p_none=torch.nn.Parameter(torch.ones(1));p_zero=torch.nn.Parameter(torch.ones(1))
        probe=torch.optim.AdamW([p_none,p_zero],lr=.01,weight_decay=.1)
        p_zero.grad=torch.zeros_like(p_zero);probe.step()
        assert p_none.item()==1. and abs(p_zero.item()-.999)<1e-7
        # Compiled activation checkpointing must preserve both mixed skip gradients.
        check=Model(c,checkpoint=True,variant=variant)
        check.load_weights(list(trainer.params.items()), strict=True)
        ct=Trainer(check,1,2,compiled=True,chunk_size=7)
        cv,cg=ct.grad(ct.params,ids)
        bv,bg=trainer.grad(trainer.params,ids)
        checkpoint_grad_diff=max(float(np.max(np.abs(np.array(cg[n])-np.array(bg[n])))) for n in cg)
        assert abs(float(cv.item())-float(bv.item()))<2e-5 and checkpoint_grad_diff<5e-5
        row=dict(variant=variant,status="passed",parameters=sum(p.numel() for p in ref.parameters()),
                 logits_max_abs_diff=logits_diff,loss_abs_diff=loss_diff,grad_max_abs_diff=grad_diff,
                 masked_position_logits_max_abs_diff=masked_diff,steps=10,curve=curve,
                 curve_max_loss_abs_diff=curve_diff,final_weights_max_abs_diff=weights_diff,
                 unused_parameters=unused,unused_mlx_gradients="zero",unused_adamw="skipped, including decay",
                 checkpoint_grad_max_abs_diff=checkpoint_grad_diff)
        records.append(row)
        print(json.dumps(row),flush=True)
    assert len({r['parameters'] for r in records})==1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="passed",device=device,precision="fp32",model_config=asdict(c),records=records),indent=2)+'\n')


if __name__=='__main__':
    main()
