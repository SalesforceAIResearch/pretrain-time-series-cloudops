from typing import Optional, Iterator

import numpy as np
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets.distributed import split_dataset_by_node
from gluonts.transform import Transformation
from gluonts.dataset.common import DataEntry
from gluonts.dataset.split import DateSplitter, OffsetSplitter
from torch.distributed import get_rank, get_world_size
from torch.utils.data import IterableDataset, get_worker_info

from .transform.convert import ProcessDataEntryTransform


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        process_data_entry: ProcessDataEntryTransform,
        transformation: Transformation,
        splitter: Optional[DateSplitter | OffsetSplitter] = None,
        is_train: bool = False,
        sample: Optional[str] = None,
        seed: Optional[int] = None,
        num_batches_per_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.process_data_entry = process_data_entry
        self.transformation = transformation
        self.splitter = splitter
        self.is_train = is_train
        self.sample = sample
        self.generator = np.random.default_rng(seed)
        self.num_batches_per_epoch = num_batches_per_epoch
        self.batch_size = batch_size

        shard_dataset = (
            ((is_train and sample is None) or not is_train)
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

        if shard_dataset:
            self.dataset = split_dataset_by_node(
                dataset,
                rank=get_rank(),
                world_size=get_world_size(),
            )

        self.probabilities = self.get_probabilities()

    @staticmethod
    def worker_init_fn(worker_id: int):
        worker_info = get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % (2**32 - 1))
            dataset = worker_info.dataset
            dataset.generator = np.random.default_rng(worker_info.seed)

            dataset.num_batches_per_epoch = (
                dataset.num_batches_per_epoch
                if dataset.num_batches_per_epoch is None
                else dataset.num_batches_per_epoch // worker_info.num_workers
                if worker_id != worker_info.num_workers - 1
                else (
                    dataset.num_batches_per_epoch // worker_info.num_workers
                    + dataset.num_batches_per_epoch % worker_info.num_workers
                )
            )

    @staticmethod
    def get_length(example: dict[str, any]) -> dict[str, any]:
        target = example["target"]
        if isinstance(target[0], list):
            length = len(target[0])
        else:
            length = len(target)
        example["length"] = length
        return example

    def get_probabilities(self) -> Optional[np.ndarray]:
        if self.sample == "uniform":
            probabilities = np.asarray([1 / len(self.dataset)] * len(self.dataset))
        elif self.sample == "proportional":
            lengths = np.asarray(self.dataset.map(self.get_length)["length"])
            probabilities = lengths / lengths.sum()
        else:
            probabilities = None
        return probabilities

    def __iter__(self) -> Iterator[DataEntry]:
        shuffle = self.is_train and self.sample is not None
        dataset = self.data_iterator(self.dataset, self.indices_iterator(shuffle))
        dataset = self.process_data_entry.apply(dataset)
        if self.splitter is not None:
            dataset, _ = self.splitter.split(dataset)
        yield from self.transformation.apply(dataset, is_train=self.is_train)

    def indices_iterator(self, shuffle: bool = False):
        if shuffle:
            for i in range(self.num_batches_per_epoch * self.batch_size):
                yield self.generator.choice(len(self.dataset), p=self.probabilities)
        else:
            length = len(self.dataset)
            worker_info = get_worker_info()
            if worker_info is not None:
                start = worker_info.id * (length // worker_info.num_workers)
                end = (worker_info.id + 1) * (length // worker_info.num_workers)
                if worker_info.id == worker_info.num_workers - 1:
                    end += length % worker_info.num_workers
            else:
                start = 0
                end = length
            for i in range(start, end):
                yield i

    @staticmethod
    def data_iterator(dataset, indices_iterator) -> Iterator[DataEntry]:
        for i in indices_iterator:
            yield dataset[i]
