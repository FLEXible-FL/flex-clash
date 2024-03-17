"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from flex.data import Dataset


def _indentity_poisoning(x, y):
    return x, y


@dataclass(frozen=True)
class PoisonedDataset(Dataset):
    """
    Dataset class that allows to apply a poison function lazily. You are not
    supposed to use this class directly, but to use the `data_poisoner` decorator.
    """

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
        return (self.poisoning_function(x, y) for (x, y) in super().__iter__())

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
