from __future__ import annotations

import torch

from urban_act.checkpoints import restore_rng_state


class FakeMappedState:
    def __init__(self, cpu_value: torch.Tensor) -> None:
        self.cpu_value = cpu_value
        self.detached = False

    def detach(self) -> FakeMappedState:
        self.detached = True
        return self

    def cpu(self) -> torch.Tensor:
        return self.cpu_value


def test_restore_rng_state_moves_mapped_torch_state_to_cpu(monkeypatch) -> None:
    expected = torch.arange(8, dtype=torch.uint8)
    mapped = FakeMappedState(expected)
    restored: list[torch.Tensor] = []
    monkeypatch.setattr(torch, "set_rng_state", restored.append)

    restore_rng_state({"torch_rng_state": mapped})

    assert mapped.detached is True
    assert restored == [expected]


def test_restore_rng_state_moves_mapped_cuda_states_to_cpu(monkeypatch) -> None:
    expected = [torch.arange(4, dtype=torch.uint8), torch.arange(5, dtype=torch.uint8)]
    mapped = [FakeMappedState(state) for state in expected]
    restored: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)

    restore_rng_state({"cuda_rng_state": mapped})

    assert all(state.detached for state in mapped)
    assert restored == [expected]
