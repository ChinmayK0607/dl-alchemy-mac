"""Lifecycle/queue/summary verification. All real W&B runs are offline."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from .config import make_config, run_name
from .model import Config, Model
from .queue import execute, parser as queue_parser
from .run import train
from .tracking import Tracking, code_hash
from .train_mlx import Trainer
from .runtime import configure_test_device


class Fixture:
    metadata = {"synthetic_test_fixture":True}
    def __init__(self, rows):
        self.array=np.tile(np.arange(1024,dtype=np.uint16)%128,(rows,1))
    def __len__(self):
        return len(self.array)
    def __getitem__(self,key):
        return self.array[key]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("results/T4_tests.json"))
    args=p.parse_args()
    Path('.build/tmp').mkdir(parents=True, exist_ok=True)
    device = configure_test_device(mx)
    records=[]
    def record(test,**kw):
        records.append(dict(test=test,status="passed",**kw))
        print(json.dumps(records[-1]),flush=True)
    plan=json.loads(Path('plans/popquiz.json').read_text())
    assert len(plan)==37
    assert len({(r['variant'],r['learning_rate']) for r in plan})==37
    assert [p['learning_rate'] for p in plan]==[lr for lr in (.003,.001,.01,.0003,.03) for _ in range(7)]+[.0001,.0001]
    assert all(p['preset']=='local' for p in plan)
    assert all(sum(r['variant']==v and r['question_group']=='q1' for r in plan)==5
               for v in ('reference','no_prenorm','no_residuals','no_mlp','no_attention','no_rope'))
    record('plan',runs=37,unique=True)
    c=make_config('local',dict(num_train_sequences=128,precision='fp32',num_evals=1))
    assert c.variant=='reference' and 'reference' in run_name(c)
    model=Model(Config.tiny())
    model.initialize(42)
    trainer=Trainer(model,1,2,compiled=True)
    trainer.materialize()
    initial={n:np.array(v).copy() for n,v in trainer.params.items()}
    calls=[]
    def nan_grad(params,ids):
        calls.append(1)
        return mx.array(float('nan')), {n:mx.zeros_like(v) for n,v in params.items()}
    trainer.grad=nan_grad
    value,norm=trainer.step(mx.array(np.zeros((2,8),dtype=np.int32)))
    assert math.isnan(float(value.item())) and len(calls)==1 and trainer.step_number==0
    assert trainer.last_step_tokens==8
    for n,v in trainer.params.items():
        np.testing.assert_array_equal(np.array(v),initial[n])
    record('nonfinite_microbatch_stops_before_update',microbatches_executed=1,optimizer_updates=0)
    with tempfile.TemporaryDirectory(dir='.build/tmp',prefix='t4_') as tmp:
        root=Path(tmp)
        with patch('mlx_port.run.load_datasets',return_value=(Fixture(128),Fixture(17))), \
             patch('mlx_port.model.depth_model_config',return_value=Config.tiny()):
            s=train(c,device='cpu',output_dir=root/'normal',max_steps=2,verification_context=8,
                    verification_val_sequences=9,wandb='offline')
            path=Path(s['run_dir'])
            assert s['status']=='verification_complete' and math.isfinite(s['final_val_loss'])
            assert not (path/'final.safetensors').exists()
            assert s['wandb_mode']=='offline' and not s['wandb_error'],s
            config=json.loads((path/'config.json').read_text())
            assert config['code_hash']==code_hash() and len(config['code_hash'])==64
            assert all(k in config for k in ('variant','learning_rate','preset','model_config'))
            assert list(path.glob('wandb/offline-run-*'))
            record('real_offline_wandb_no_default_weights',steps=s['steps'],final_val_loss=s['final_val_loss'])
            # A nonfinite training loss must stop before eval and finish tracking.
            with patch.object(Trainer,'step_micro_batches',return_value=(mx.array(float('nan')),mx.array(0.))) as step, \
                 patch('mlx_port.run.evaluate',side_effect=AssertionError('Must not evaluate a diverged run')):
                s=train(c,device='cpu',output_dir=root/'diverged',max_steps=2,verification_context=8,
                        verification_val_sequences=9,wandb='offline',save_weights=True)
                assert s['status']=='diverged' and math.isnan(s['final_val_loss']) and s['steps']==1
                assert step.call_count==1
                path=Path(s['run_dir'])
                assert not (path/'final.safetensors').exists()
                assert len((path/'metrics.jsonl').read_text().splitlines())==1
                assert json.loads((path/'summary.json').read_text())['status']=='diverged'
            with patch.object(Trainer,'step_micro_batches',side_effect=RuntimeError('injected failure')):
                s=train(c,device='cpu',output_dir=root/'error',max_steps=2,verification_context=8,
                        verification_val_sequences=9,wandb='disabled')
                assert s['status']=='error' and 'injected failure' in s['error']
            record('divergence_and_error_finalization',diverged_steps=1,weights_saved=False)

        # Mock the SDK to test online fallback without an online call or network.
        calls=[]
        class FakeRun:
            summary={}
            def log(self,*a,**kw): pass
            def finish(self,**kw): calls.append('finish')
        def init(**kw):
            calls.append(kw['mode'])
            if kw['mode']=='online': raise RuntimeError('simulated init failure')
            return FakeRun()
        fake=SimpleNamespace(init=init,Settings=lambda **kw:kw)
        with patch.dict(sys.modules,{'wandb':fake}):
            t=Tracking('online',root,dict(run_name='test',variant='reference',learning_rate=.003),'q1')
            assert t.mode=='offline'
            t.finish(dict(status='diverged',final_val_loss=float('nan')))
        assert calls==['online','offline','finish']
        record('mocked_online_fallback',actual_online_runs=0)

        # Real child processes: invalid config, valid completion, then restart skip.
        plan_path=root/'plan.json'
        plan_path.write_text(json.dumps([
            dict(preset='local',variant='bad',question_group='q1'),
            dict(preset='local',variant='no_attention',learning_rate=.003,question_group='q1')]))
        qa=queue_parser().parse_args(['--plan',str(plan_path),'--device','cpu','--wandb','disabled',
            '--output-dir',str(root/'queue'),'--results',str(root/'queue.jsonl'),
            '--token-budget-sequences','128','--max-steps','1','--verification-context','8',
            '--verification-val-sequences','9','--timeout','120'])
        rows=execute(qa)
        assert [r['status'] for r in rows]==['error','verification_complete']
        assert rows[1]['config']['num_train_sequences']==128
        again=execute(qa)
        assert again[1]['skipped'] and again[1]['status']=='verification_complete'
        # Inject a timeout, confirm next subprocess still launches and succeeds.
        plan_path.write_text(json.dumps([
            dict(preset='local',variant=v,learning_rate=.001,question_group='q1') for v in ('no_mlp','reference')]))
        real_run=subprocess.run
        count=0
        def timeout_once(cmd,**kw):
            nonlocal count
            count+=1
            if count==1: raise subprocess.TimeoutExpired(cmd,kw['timeout'])
            return real_run(cmd,**kw)
        with patch('mlx_port.queue.subprocess.run',side_effect=timeout_once):
            rows=execute(qa)
        assert [r['status'] for r in rows]==['timeout','verification_complete']
        assert (Path(rows[0]['run_dir'])/'summary.json').exists()
        record('isolated_queue_failure_timeout_resume',statuses=['error','verification_complete','timeout','verification_complete'])

        # Actual timeout termination (no training): subprocess.run must reap child.
        try:
            real_run([sys.executable,'-c','import time; time.sleep(10)'],timeout=.05,capture_output=True)
            raise AssertionError('Timeout did not fire')
        except subprocess.TimeoutExpired:
            pass
        record('actual_subprocess_timeout',timeout_seconds=.05)
        # Synthetic summary values are isolated in this temporary directory.
        from .popquiz_summary import summarize
        synthetic=[]
        for v,lr,val,status in [('reference',.003,4.5,'complete'),('reference',.0001,4.2,'complete'),
                                ('mixed_residual',.001,4.1,'complete'),('no_mlp',.03,float('nan'),'diverged')]:
            path=root/f'{v}-{lr}'
            path.mkdir()
            (path/'metrics.jsonl').write_text(json.dumps(dict(step=1,tokens=65536,val_loss=val))+'\n')
            synthetic.append(dict(run_dir=str(path),status=status,final_val_loss=val,
                                  config=asdict(replace(c,variant=v,learning_rate=lr))))
        synthetic.append(synthetic[0])
        synthetic.append(dict(run_dir='verification',status='verification_complete',final_val_loss=.1,config=asdict(c)))
        sf=root/'synthetic.jsonl'
        sf.write_text(''.join(json.dumps(r)+'\n' for r in synthetic))
        s=summarize(sf,root/'plots')
        assert s['runs']==4 and abs(s['delta_loss']+.1)<1e-6 and s['q1_ranking']==['reference']
        assert s['best']['reference']==.0001
        report=(root/'plots/popquiz_summary.md').read_text()
        assert 'non-finite' in report
        for name in ('popquiz_loss_vs_lr.png','popquiz_best_curves.png'):
            assert (root/'plots'/name).stat().st_size>10000
        record('summary_dedup_q1_q2_and_plots',synthetic=True,delta_loss=s['delta_loss'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='passed',device=device,records=records),indent=2)+'\n')


if __name__=='__main__':
    main()
