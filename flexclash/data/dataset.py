from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from flex.data import Dataset


def _indentity_poisoning(x, y):
    return x, y


@dataclass(kw_only=True, frozen=True)
class PoisonedDataset(Dataset):
    poisoning_function: Callable = field(init=True, default=_indentity_poisoning)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if isinstance(item, tuple):
            (x, y) = item
            return self.poisoning_function(x, y)
        elif isinstance(item, Dataset):
            return PoisonedDataset(
                X_data=item.X_data,
                y_data=item.y_data,
                poisoning_function=self.poisoning_function,
            )
        else:
            raise ValueError("The item is not a tuple neither a Dataset")

    def __iter__(self):
        return (
            self.poisoning_function(x, y)
            for (x, y) in zip(
                self.X_data,
                self.y_data if self.y_data is not None else [None] * len(self),
            )
        )

    def to_numpy(self, x_dtype=None, y_dtype=None):
        """Function to return the FlexDataObject as numpy arrays."""
        (features, labels) = super().to_list()
        return (np.array(features, dtype=x_dtype), np.array(labels, dtype=y_dtype))

    def to_list(self):
        assert self.y_data is not None, "PoisonedDataset must have labels."
        arrays = super().to_list()
        features = []
        labels = []
        for feature, label in zip(*arrays):
            new_feat, new_label = self.poisoning_function(feature, label)
            features.append(new_feat)
            labels.append(new_label)
        return features, labels
