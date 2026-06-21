import logging

import torch.distributed as dist


def is_rank_0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def log_for_0(msg, *args, level=logging.INFO):
    if not is_rank_0():
        return
    logging.log(level, msg, *args)
