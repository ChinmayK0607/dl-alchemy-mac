"""The reference schedulers, indexed by completed optimizer steps."""
import math
import struct


def learning_rate(step, total_steps=9375, base=3e-3, schedule="linear", warmup_percent=0.01):
    warmup = int(total_steps * warmup_percent)
    if schedule in ("constant", "const", "flat"):
        return base  # Reference ConstantScheduler ignores warmup.
    if schedule not in ("linear", "lin", "cos") and not schedule.startswith("wsd"):
        raise ValueError(f"Unknown lr_schedule: {schedule}")
    if schedule.startswith("wsd"):
        decay = int(total_steps * float(schedule[3:]))
    if step < warmup:
        return base * step / warmup
    if schedule.startswith("wsd"):
        stable_end = total_steps - decay
        if step < stable_end:
            return base
        progress = (step - stable_end) / max(1, decay)
    else:
        progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    if schedule == "cos":
        # Reference evaluates torch.cos(torch.tensor(progress * torch.pi)) in
        # fp32. Reproduce that rounding without a runtime PyTorch dependency.
        f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
        return base * 0.5 * (1 + f32(math.cos(f32(progress * math.pi))))
    return base * (1 - progress)
