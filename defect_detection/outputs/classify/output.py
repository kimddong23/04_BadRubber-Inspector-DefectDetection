from dataclasses import dataclass
from typing import List, Iterator, Tuple, Optional

# ---------------------------------
# Single region classification
# ---------------------------------

@dataclass(frozen=True)
class Classification:
    class_id: int
    class_name: str
    confidence: float
    is_pass: bool
    color: Tuple[int, int, int]

# ---------------------------------
# Per-image batch item
# ---------------------------------

@dataclass(frozen=True)
class ClassificationBatchItem:
    regions: List[Classification]

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)


# ---------------------------------
# Batch output
# ---------------------------------

@dataclass
class RegionClassificationOutput:
    __slots__ = ("batch",)

    batch: List[List[Classification]]  # [B][R]

    def __len__(self):
        return len(self.batch)

    def __iter__(self) -> Iterator[ClassificationBatchItem]:
        for regions in self.batch:
            yield ClassificationBatchItem(regions)

    def __getitem__(self, idx: int) -> ClassificationBatchItem:
        return ClassificationBatchItem(self.batch[idx])

def merge_cls_outputs(
    inputs: List[Optional[RegionClassificationOutput]],
) -> Optional[RegionClassificationOutput]:

    inputs = [x for x in inputs if x is not None]

    if len(inputs) == 0:
        return None

    batch_size = len(inputs[0].batch)

    for inp in inputs:
        if len(inp.batch) != batch_size:
            raise ValueError("Batch size mismatch in RegionClassificationOutput merge")

    merged_batch: List[List[Classification]] = []

    for b in range(batch_size):
        merged_regions: List[Classification] = []

        for inp in inputs:
            merged_regions.extend(inp.batch[b])

        merged_batch.append(merged_regions)

    return RegionClassificationOutput(merged_batch)