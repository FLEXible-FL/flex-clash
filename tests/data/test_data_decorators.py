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
import unittest

import numpy as np
import pytest
from flex.data import Dataset, FedDataset

from flexclash.data import data_poisoner


@pytest.fixture(name="fld")
def fixture_flex_dataset():
    """Function that returns a FlexDataset provided as example to test functions.

    Returns
    -------
        FedDataset: A FlexDataset generated randomly
    """
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd = Dataset.from_array(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd1 = Dataset.from_array(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd2 = Dataset.from_array(X_data, y_data)
    return FedDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})


class TestPoisoningDecorators(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_flex_dataset(self, fld):
        self._fld = fld

    def test_data_poisoner_decorator(self):
        @data_poisoner
        def change_labels_to_one(feature, label):
            return feature, 1

        poisoned_fld = self._fld.apply(change_labels_to_one, node_ids=["client_1"])
        _, poisoned_labels = poisoned_fld["client_1"].to_list()

        assert all(poisoned_labels == np.ones_like(poisoned_labels))

    def test_data_poisoner_insuficient_return_values(self):
        @data_poisoner
        def change_labels_to_one(feature, label):
            return feature

        with pytest.raises(ValueError):
            poisoned_fld = self._fld.apply(change_labels_to_one, node_ids=["client_1"])
            poisoned_fld["client_1"].to_list()

    def test_data_poisoner_composition_works(self):
        @data_poisoner
        def change_labels_to_one(feature, label):
            return feature, 1

        @data_poisoner
        def add_one_to_label(feature, label):
            return feature, label + 1

        poisoned_fld = self._fld.apply(change_labels_to_one, node_ids=["client_1"])
        poisoned_fld = poisoned_fld.apply(add_one_to_label, node_ids=["client_1"])
        _, poisoned_labels = poisoned_fld["client_1"].to_list()

        assert all(poisoned_labels == 2 * np.ones_like(poisoned_labels))
