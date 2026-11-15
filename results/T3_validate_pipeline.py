"""Task-local orchestration; wait for bounded download, then validate CPU artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path.cwd()
ready=root/'data/dclm/train_ds42/metadata.json'
if not ready.exists():
    raise FileNotFoundError("Run scripts/setup.sh before this historical validation pipeline")
env=os.environ.copy()
(root/'.build/tmp').mkdir(parents=True,exist_ok=True)
env['TMPDIR']=str(root/'.build/tmp')
env['MLX_DEVICE']='cpu'
env['MPLCONFIGDIR']=str(root/'.build/matplotlib')
commands=[]
for preset in ['smoke','local']:
    commands.append((f'T3_cpu_{preset}', ['-m','mlx_port.run','--preset',preset,'--device','cpu',
                     '--max-steps','3','--verification-context','32','--verification-val-sequences','16',
                     '--output-dir','runs/T3-cpu']))
commands.append(('T3_tests',['-m','mlx_port.test_t3','--verify-source']))
commands.append(('T3_baseline',['-m','mlx_port.baseline']))
configs=[{'dropout':1},
         {'learning_rate':.001,'lr_schedule':'cos','dropout':.1,'run_name_suffix':'cpu-grid'},
         {'learning_rate':.003,'lr_schedule':'wsd0.1','dropout':0.,'run_name_suffix':'cpu-grid'}]
(root/'results/T3_cpu_grid.json').write_text(json.dumps(configs,indent=2)+'\n')
commands.append(('T3_cpu_sweep',['-m','mlx_port.sweep','--preset','local','--device','cpu',
                 '--configs','results/T3_cpu_grid.json','--max-steps','2','--verification-context','32',
                 '--verification-val-sequences','16','--output-dir','runs/T3-cpu-sweep',
                 '--results','results/T3_cpu_sweep.jsonl','--timeout','120']))
commands.append(('T3_plot_check',['-m','mlx_port.plot_runs','--runs-dir','runs/T3-cpu',
                 '--include-verification','--output','results/T3_cpu_verification.png']))
for name,args in commands:
    print('START',name,flush=True)
    started=time.perf_counter()
    with (root/'results'/f'{name}.log').open('w') as f:
        proc=subprocess.run([str(root/'.venv/bin/python'),*args],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    row=dict(command=args,returncode=proc.returncode,wall_seconds=time.perf_counter()-started)
    print(json.dumps(row),flush=True)
    if proc.returncode:
        print((root/'results'/f'{name}.log').read_text()[-5000:],flush=True)
        sys.exit(proc.returncode)
print('ALL_PIPELINE_CHECKS_PASSED',flush=True)
