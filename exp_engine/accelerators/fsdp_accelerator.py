import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from .base_accelerator import BaseAccelerator


class FSDPAccelerator(BaseAccelerator):
    """
    Currently only support fully-shard.

    """

    def __init__(self, wrap_policy=None, enable_offload_params=False):
        super().__init__()

        assert torch.cuda.is_available() and self.num_processes >= 1, "FSDP requires at least one GPU"
        self.wrap_policy = wrap_policy
        self.enable_offload_params = enable_offload_params
        self.mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        self.dtype = self.mixed_precision.param_dtype

        # init global process group
        dist.init_process_group(backend="nccl", init_method="env://")
        self.device_mesh = self._init_process_groups()

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, model):
        return model.module

    def print(self, *args, **kwargs):
        if self.process_index == 0:
            print(*args, **kwargs)

    def reduce_sum(self, tensor):
        world_size = self.num_processes
        if world_size < 2:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _init_process_groups(self):
        """
        Get the process groups for shard and replicate.

        Current implementation only support fully-shard.
        """
        return init_device_mesh("cuda", mesh_shape=(self.dist_cfg["world_size"],), mesh_dim_names=["fsdp"])

    def prepare(self, model, optimizer=None):
        raise NotImplementedError(
            "FSDP does not support prepare method because FSDP must prepare model first and then prepare optimizer(use prepared model). Please use prepare_model and prepare_optimizer instead."
        )

    def prepare_model(self, model):
        if self._model is None:
            self._model = FSDP(
                module=model,
                use_orig_params=True,
                auto_wrap_policy=self.wrap_policy,
                device_id=self.process_index,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=self.mixed_precision,
                cpu_offload=CPUOffload(offload_params=self.enable_offload_params),
                device_mesh=self.device_mesh,
            )
        return self._model

    def wait_for_everyone(self):
        if self.num_processes < 2:
            return
        dist.barrier(device_ids=[self.local_process_index])

    def prepare_optimizer(self, optimizer):
        self._optimizer = optimizer
        return self._optimizer

    def offload_states(self):
        raise NotImplementedError

    def reload_states(self):
        raise NotImplementedError
