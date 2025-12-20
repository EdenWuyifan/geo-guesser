from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def rank() -> int:
    return _int_env("RANK", 0)


def world_size() -> int:
    return _int_env("WORLD_SIZE", 1)


def local_rank() -> int:
    return _int_env("LOCAL_RANK", 0)


def is_distributed() -> bool:
    return world_size() > 1


def is_main_process() -> bool:
    return rank() == 0


@dataclass(frozen=True)
class DistState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def main(self) -> bool:
        return self.rank == 0


def init_distributed(requested_device: str = "cuda") -> DistState:
    """
    Initializes torch.distributed when launched via torchrun.

    - Reads RANK/WORLD_SIZE/LOCAL_RANK.
    - Picks NCCL on CUDA, else GLOO.
    - Sets the current CUDA device to LOCAL_RANK.
    - Returns DistState with the device to use.
    """
    r = rank()
    ws = world_size()
    lr = local_rank()

    if ws > 1:
        import torch.distributed as dist

        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

        if torch.cuda.is_available() and requested_device.startswith("cuda"):
            torch.cuda.set_device(lr)
            device = torch.device("cuda", lr)
        else:
            device = torch.device("cpu")

        return DistState(rank=r, world_size=ws, local_rank=lr, device=device)

    # Single-process
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    return DistState(rank=r, world_size=ws, local_rank=lr, device=device)


def barrier(state: DistState) -> None:
    if not state.distributed:
        return
    import torch.distributed as dist

    dist.barrier()


def all_reduce_sum_(tensor: torch.Tensor, state: DistState) -> torch.Tensor:
    if not state.distributed:
        return tensor
    import torch.distributed as dist

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_object(obj: object, state: DistState) -> list[object]:
    if not state.distributed:
        return [obj]
    import torch.distributed as dist

    gathered: list[object] = [None] * state.world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def destroy_process_group(state: DistState) -> None:
    if not state.distributed:
        return
    import torch.distributed as dist

    dist.destroy_process_group()

