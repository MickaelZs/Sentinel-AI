"""Chronological train, validation, and test splitting."""

from dataclasses import dataclass

import pandas as pd

from sentinel_ai.ml.features import TARGET_COLUMN


@dataclass(frozen=True)
class TemporalSplit:
    """Non-overlapping chronological dataset partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(dataset: pd.DataFrame) -> TemporalSplit:
    """Split rows by occurred_at into 70% train, 15% validation, and 15% test."""
    ordered = dataset.sort_values("occurred_at", ignore_index=True)
    train_end = int(len(ordered) * 0.70)
    validation_end = train_end + int(len(ordered) * 0.15)
    split = TemporalSplit(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )
    _validate_split(split, len(dataset))
    return split


def _validate_split(split: TemporalSplit, original_rows: int) -> None:
    """Assert core split invariants before model fitting."""
    if len(split.train) + len(split.validation) + len(split.test) != original_rows:
        raise ValueError("Temporal split does not preserve all rows.")
    if not (
        split.train["occurred_at"].max() <= split.validation["occurred_at"].min()
        and split.validation["occurred_at"].max() <= split.test["occurred_at"].min()
    ):
        raise ValueError("Temporal split partitions are not chronologically ordered.")
    if any(
        partition[TARGET_COLUMN].nunique() < 2
        for partition in (split.train, split.validation, split.test)
    ):
        raise ValueError("Each temporal partition must contain both target classes.")
