"""Small W&B adapter; local artifacts remain authoritative if tracking fails."""
import hashlib
from pathlib import Path
import warnings


def code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


class Tracking:
    def __init__(self, mode, directory, config, group):
        self.run = None
        self.mode = mode
        self.error = None
        if mode == "disabled":
            return
        try:
            import wandb
            kwargs = dict(project="dl-alchemy-popquiz", name=config["run_name"],
                          dir=str(directory), config=config, group=group,
                          tags=[config["variant"], str(config["learning_rate"])],
                          settings=wandb.Settings(init_timeout=20))
            try:
                self.run = wandb.init(mode=mode, **kwargs)
            except Exception as exc:
                if mode != "online":
                    raise
                warnings.warn(f"Online W&B init failed; falling back to offline: {exc}")
                self.mode = "offline"
                self.run = wandb.init(mode="offline", **kwargs)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"W&B unavailable; preserving local metrics: {self.error}")

    def log(self, row):
        if self.run is not None:
            try:
                self.run.log(row, step=row["step"])
            except Exception as exc:
                self.error = str(exc)
                warnings.warn(f"W&B log failed: {exc}")

    def finish(self, summary):
        if self.run is not None:
            try:
                self.run.summary.update(summary)
            except Exception as exc:
                self.error = str(exc)
            finally:
                try:
                    self.run.finish(exit_code=0 if summary["status"] in ("complete", "verification_complete", "diverged") else 1)
                except Exception as exc:
                    self.error = str(exc)
