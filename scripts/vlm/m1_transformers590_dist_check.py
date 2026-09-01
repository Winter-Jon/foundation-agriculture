from __future__ import annotations

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    probe = torch.tensor([rank], device=f'cuda:{rank}')
    dist.all_reduce(probe)
    print(f'rank={rank} world={dist.get_world_size()} sum={probe.item()} device={torch.cuda.current_device()}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
