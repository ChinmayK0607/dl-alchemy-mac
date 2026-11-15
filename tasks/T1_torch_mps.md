# T1: fastest PyTorch-MPS port of the A1 default run

Read tasks/COMMON.md first.

Write torch_mps/ in the workspace:
1. torch_mps/train_mps.py: a training step that uses the reference model code (import from the reference repo via sys.path;
   copy-and-patch into torch_mps/ only where MPS needs it) on device "mps", with the reference loss and AdamW settings.
   Random token batches are fine for throughput.
2. Correctness: on CPU-fp32 vs MPS (same seed/weights), show logits/loss at init match closely and that the loss after
   ~20 steps on a fixed repeated batch decreases similarly.
3. Optimize. Try at least: fp32 vs bf16 autocast vs fp16 autocast vs pure bf16 weights (keeping fp32 master if needed);
   micro-batch sizes (8/16/32/64) with grad accumulation to 64; SDPA is_causal path; torch.compile on MPS if it works;
   fused AdamW (foreach/fused flags) ; float32 matmul precision settings; channels/contiguity of the loss; avoiding the
   fp32 cast of 64x1024x4096 logits if possible (chunked CE). Keep what helps.
4. You cannot measure GPU speed yourself: implement each option behind a flag and make torch_mps/bench.py sweep them. Benchmark script torch_mps/bench.py printing a JSON line per config; save to results/torch_mps.jsonl.

Report: tasks/T1.report.md
