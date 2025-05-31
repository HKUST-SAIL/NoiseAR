import abc
import os
from abc import abstractmethod
from functools import cached_property

import torch

__all__ = ["BaseAccelerator"]


class BaseAccelerator(abc.ABC):
    """
    basic accelerator, provide basic functions for distributed training.
    """

    def __init__(self) -> None:
        self._model = None
        self._optimizer = None

        self.process_index = self.dist_cfg["rank"]
        self.num_processes = self.dist_cfg["world_size"]
        self.local_process_index = self.dist_cfg["local_rank"]

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_process_index)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @cached_property
    def dist_cfg(self) -> dict:
        return {
            "rank": int(os.getenv("RANK", 0)),
            "world_size": int(os.getenv("WORLD_SIZE", 1)),
            "local_rank": int(os.getenv("LOCAL_RANK", 0)),
            "local_world_size": int(os.getenv("LOCAL_WORLD_SIZE", 1)),
            "master_addr": os.getenv("MASTER_ADDR", "127.0.0.1"),
            "master_port": int(os.getenv("MASTER_PORT", 12345)),
        }

    @abstractmethod
    def _init_process_groups(self) -> None:
        pass

    @property
    def model(self):  # type: ignore[no-untyped-def]
        return self._model

    @abstractmethod
    def unwrap_model(self, model):  # type: ignore[no-untyped-def]
        pass

    @property
    def optimizer(self):  # type: ignore[no-untyped-def]
        return self._optimizer

    @abstractmethod
    def prepare(self, model, optimizer=None):  # type: ignore[no-untyped-def]
        pass

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        pass

    @abstractmethod
    def wait_for_everyone(self) -> None:
        pass

    @abstractmethod
    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def print(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    @property
    def is_main_process(self):  # type: ignore[no-untyped-def]
        """True for one process per server."""
        return self.process_index == 0

    @property
    def is_local_main_process(self):  # type: ignore[no-untyped-def]
        """True for one process per server."""
        return self.local_process_index == 0

    @property
    def is_last_process(self):  # type: ignore[no-untyped-def]
        return self.process_index == self.num_processes - 1
