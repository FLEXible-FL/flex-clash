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
    fcd = Dataset.from_numpy(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd1 = Dataset.from_numpy(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd2 = Dataset.from_numpy(X_data, y_data)
    return FedDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})


class TestPoisoningDecorators(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_flex_dataset(self, fld):
        self._fld = fld

    def test_data_poisoner_decorator(self):
        @data_poisoner
        def change_labels_to_one(feature, label):
            return feature, 1

        poisoned_fld = self._fld.apply(change_labels_to_one, clients_ids=["client_1"])
        poisoned_labels = poisoned_fld["client_1"].y_data.tolist()

        assert all(poisoned_labels == np.ones_like(poisoned_labels))

    def test_data_poisoner_insuficient_return_values(self):
        @data_poisoner
        def change_labels_to_one(feature, label):
            return feature

        with pytest.raises(ValueError):
            self._fld.apply(change_labels_to_one, clients_ids=["client_1"])
