from types import SimpleNamespace

import pytest
import torch

from tabicl.train._run import Trainer


def _trainer(checkpoint_dir):
    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(checkpoint_dir=str(checkpoint_dir))
    trainer.model_config = {"row_identity_mode": "rope"}
    trainer.raw_model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.raw_model.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _step: 1.0
    )
    trainer.curr_step = 1
    return trainer


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    checkpoint_path = tmp_path / "step-1.ckpt"
    checkpoint_path.write_bytes(b"previous-complete-checkpoint")

    def fail_after_partial_write(_checkpoint, path):
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise OSError("simulated ENOSPC")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated ENOSPC"):
        trainer.save_checkpoint("step-1.ckpt")

    assert checkpoint_path.read_bytes() == b"previous-complete-checkpoint"
    assert list(tmp_path.glob(".step-1.ckpt.*.tmp")) == []
