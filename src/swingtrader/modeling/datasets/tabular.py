"""Framework-neutral tabular adapters for temporal dataset bundles."""

from dataclasses import dataclass

import pandas as pd

from swingtrader.modeling.datasets.temporal import TemporalDatasetBundle


@dataclass(frozen=True, slots=True)
class TabularDataset:
    """Own feature matrix, selected target, and aligned sample metadata."""

    X: pd.DataFrame
    y: pd.Series
    samples: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.X.index.equals(self.y.index) or not self.X.index.equals(self.samples.index):
            raise ValueError("Tabular dataset components must use identical sample indexes.")
        object.__setattr__(self, "X", self.X.copy(deep=True))
        object.__setattr__(self, "y", self.y.copy(deep=True))
        object.__setattr__(self, "samples", self.samples.copy(deep=True))


def to_tabular_dataset(bundle: TemporalDatasetBundle) -> TabularDataset:
    """Select ``X`` and ``y`` without splitting, imputing, or coercing dtypes."""
    target_column = bundle.manifest.spec.task.target_column
    return TabularDataset(
        X=bundle.features,
        y=bundle.targets[target_column].rename(target_column),
        samples=bundle.samples,
    )
