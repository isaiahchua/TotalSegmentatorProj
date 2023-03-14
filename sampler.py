import sys, os
import math
from typing import TypeVar, Optional, Iterator
import random
import numpy as np
import torch
import torchio as tio
import torch.distributed as dist
from torch.utils.data import Sampler

T_co = TypeVar('T_co', covariant=True)

class GroupSampler(Sampler):

    def __init__(self, group_indices: list, shuffle=False):
        self.indices = group_indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class BalancedGroupSampler(Sampler):

    def __init__(self, group_indices: dict, labels: list, batch_size: int,
                 shuffle=False):
        first_group = group_indices[labels[0]]
        second_group = group_indices[labels[1]]
        if len(first_group) > len(second_group):
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = True
        elif len(first_group) < len(second_group):
            self.larger_group = second_group
            self.smaller_group = first_group
            self.balance_group = True
        else:
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = False
        self.batch_size = batch_size

        padding_size = len(self.larger_group) % self.batch_size
        if padding_size <= len(self.larger_group):
            pad = random.sample(self.larger_group, padding_size)
            self.larger_group += pad
        assert len(self.larger_group) % self.batch_size == 0

        if self.balance_group:
            multiplication_factor = len(self.larger_group) // len(self.smaller_group)
            remainder = len(self.larger_group) % len(self.smaller_group)
            self.smaller_group = self.smaller_group * multiplication_factor + self.smaller_group[:remainder]
            assert len(self.smaller_group) == len(self.larger_group)

        self.shuffle = shuffle
        self.num_batches = (len(self.larger_group) + len(self.smaller_group)) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            larger_sample = random.sample(self.larger_group, len(self.larger_group))
            smaller_sample = random.sample(self.smaller_group, len(self.smaller_group))
        else:
            smaller_sample = copy(self.smaller_group)
            larger_sample = copy(self.larger_group)

        sorted_samples = self.sort_samples(larger_sample, smaller_sample)
        return iter(sorted_samples)

    def __len__(self):
        return self.num_batches

    def sort_samples(self, group1, group2):
        a = np.asarray(group1).reshape((-1, self.batch_size))
        b = np.asarray(group2).reshape((-1, self.batch_size))
        return np.concatenate((a,b), axis=1).flatten().tolist()

class GroupDistSampler(Sampler[T_co]):

    def __init__(self, group_ids: list,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.group_ids = group_ids
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.group_ids) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.group_ids) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.group_ids) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            sample_ids = random.sample(self.group_ids, len(self.group_ids))
        else:
            sample_ids = copy(self.group_ids)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(sample_ids)
            if padding_size <= len(sample_ids):
                sample_ids += sample_ids[:padding_size]
            else:
                sample_ids += (sample_ids * math.ceil(padding_size / len(sample_ids)))[:padding_size]
        else:
            sample_ids = sample_ids[:self.total_size]
        assert len(sample_ids) == self.total_size

        sample_ids = sample_ids[self.rank:self.total_size:self.num_replicas]
        assert len(sample_ids) == self.num_samples

        return iter(sample_ids)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

class DoubleBalancedGroupDistSampler(Sampler[T_co]):

    def __init__(self, first_group: list, second_group: list,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.group1 = first_group
        self.group2 = second_group
        if len(first_group) > len(second_group):
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = True
        elif len(first_group) < len(second_group):
            self.larger_group = second_group
            self.smaller_group = first_group
            self.balance_group = True
        else:
            self.larger_group = first_group
            self.smaller_group = second_group
            self.balance_group = False
        if self.balance_group:
            multiplication_factor = len(self.larger_group) // len(self.smaller_group)
            remainder = len(self.larger_group) % len(self.smaller_group)
            self.smaller_group = self.smaller_group * multiplication_factor + self.smaller_group[:remainder]
            assert len(self.smaller_group) == len(self.larger_group)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.larger_group) % self.num_replicas != 0:
            self.num_samples = 2 * math.ceil(
                (self.group_size - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = 2 * math.ceil(len(self.larger_group) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            larger_sample = random.sample(self.larger_group, len(self.larger_group))
            smaller_sample = random.sample(self.smaller_group, len(self.smaller_group))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size//2 - len(larger_sample)
            if padding_size <= len(larger_sample):
                smaller_sample += smaller_sample[:padding_size]
                larger_sample += larger_sample[:padding_size]
            else:
                smaller_sample += (smaller_sample * math.ceil(padding_size / len(smaller_sample)))[:padding_size]
                larger_sample += (larger_sample * math.ceil(padding_size / len(larger_sample)))[:padding_size]
        else:
            smaller_sample = smaller_sample[:self.total_size]
            larger_sample = larger_sample[:self.total_size]
        assert len(larger_sample) + len(smaller_sample) == self.total_size

        sorted_samples = self.sort_samples(larger_sample, smaller_sample)
        rank_samples = sorted_samples[self.rank:self.total_size:self.num_replicas]
        assert len(rank_samples) == self.num_samples

        return iter(rank_samples)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def sort_samples(self, group1, group2):
        a = np.asarray(group1).reshape((-1, self.num_replicas))
        b = np.asarray(group2).reshape((-1, self.num_replicas))
        return np.concatenate((a,b), axis=1).flatten().tolist()

    def sort_samples2(self, group1, group2):
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        joined = np.concatenate((np.asarray(group1), np.asarray(group2)))
        rng.shuffle(joined)
        return joined.tolist()

if __name__ == "__main__":
    pass
