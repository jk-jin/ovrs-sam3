from __future__ import annotations

from typing import Any, Dict, Iterator, List

import torch
from torch.utils.data import Sampler


class ResumableRandomBatchSampler(Sampler):
    """Batch sampler that supports checkpoint/resume with fixed per-sample augmentation seeds.

    Each sample gets a deterministic augmentation seed derived from the shuffled
    order of the current cycle. This makes augmentations reproducible regardless
    of worker assignment, prefetch order, or persistent_workers.

    ``committed_batch_index`` advances every time ``commit_batch()`` is called
    (once per consumed batch). Resume starts from the committed position without
    replaying or discarding prefetched batches.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ):
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)

        self.generator = generator

        self.cycle = 0
        self.committed_batch_index = 0

        self._sample_order: List[int] = []
        self._augmentation_seeds: List[int] = []
        self._batch_count = 0

        self._shuffle_cycle()

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._batch_count

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[tuple[int, int]]]:
        for batch_idx in range(self.committed_batch_index, self._batch_count):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(self._sample_order))
            batch = [
                (int(self._sample_order[i]), int(self._augmentation_seeds[i]))
                for i in range(start, end)
            ]
            yield batch

    # ------------------------------------------------------------------
    # Shuffle
    # ------------------------------------------------------------------

    def _shuffle_cycle(self) -> None:
        n = self.dataset_size
        order = torch.randperm(n, generator=self.generator).tolist()
        seeds = torch.randint(
            0,
            2 ** 31 - 1,
            (n,),
            generator=self.generator,
        ).tolist()

        self._sample_order = [int(x) for x in order]
        self._augmentation_seeds = [int(x) for x in seeds]

        if self.drop_last:
            self._batch_count = len(self._sample_order) // self.batch_size
        else:
            self._batch_count = (
                (len(self._sample_order) + self.batch_size - 1) // self.batch_size
            )

    # ------------------------------------------------------------------
    # Commit — called by Trainer after every consumed batch
    # ------------------------------------------------------------------

    def commit_batch(self) -> None:
        self.committed_batch_index += 1
        if self.committed_batch_index >= self._batch_count:
            self.committed_batch_index = 0
            self.cycle += 1
            self._shuffle_cycle()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cycle": int(self.cycle),
            "committed_batch_index": int(self.committed_batch_index),
            "sample_order": [int(x) for x in self._sample_order],
            "augmentation_seeds": [int(x) for x in self._augmentation_seeds],
            "generator_state": self.generator.get_state(),
            "dataset_size": int(self.dataset_size),
            "batch_size": int(self.batch_size),
            "drop_last": bool(self.drop_last),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        ds = int(state["dataset_size"])
        bs = int(state["batch_size"])
        dl = bool(state["drop_last"])

        if ds != self.dataset_size:
            raise ValueError(
                f"dataset_size mismatch: checkpoint has {ds}, "
                f"current dataloader has {self.dataset_size}"
            )
        if bs != self.batch_size:
            raise ValueError(
                f"batch_size mismatch: checkpoint has {bs}, "
                f"current dataloader has {self.batch_size}"
            )
        if dl != self.drop_last:
            raise ValueError(
                f"drop_last mismatch: checkpoint has {dl}, "
                f"current dataloader has {self.drop_last}"
            )

        self.cycle = int(state["cycle"])
        self.committed_batch_index = int(state["committed_batch_index"])
        self._sample_order = [int(x) for x in state["sample_order"]]
        self._augmentation_seeds = [int(x) for x in state["augmentation_seeds"]]

        if "generator_state" in state:
            self.generator.set_state(state["generator_state"])

        if self.drop_last:
            self._batch_count = len(self._sample_order) // self.batch_size
        else:
            self._batch_count = (
                (len(self._sample_order) + self.batch_size - 1) // self.batch_size
            )
