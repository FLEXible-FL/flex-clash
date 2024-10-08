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
import copy
import unittest

import pytest
from flex.data import Dataset, FedDataDistribution
from flex.model import FlexModel
from flex.pool import FlexPool, deploy_server_model, init_server_model
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from flexclash.model import model_poisoner


class TestModelPoisoningDecorators(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_iris_dataset(self):
        iris = load_iris()
        c_iris = Dataset.from_array(iris.data, iris.target)
        self.f_iris = FedDataDistribution.iid_distribution(c_iris, n_nodes=5)

    def test_decorators(self):
        @init_server_model
        def build_server_model():
            flex_model = FlexModel()
            flex_model["model"] = KNeighborsClassifier(n_neighbors=3)
            return flex_model

        @deploy_server_model
        def copy_server_model_to_clients(server_flex_model: FlexModel):
            return copy.deepcopy(server_flex_model)

        @model_poisoner
        def poison_model(client_model: FlexModel):
            client_model["model"] = KNeighborsClassifier(n_neighbors=6)
            return client_model

        @model_poisoner
        def bad_poison_model(client_model: FlexModel):
            client_model["model"] = KNeighborsClassifier(n_neighbors=6)

        p = FlexPool.client_server_pool(self.f_iris, init_func=build_server_model)
        p.servers.map(copy_server_model_to_clients, p.clients)

        poisoned_clients = p.clients.select(2)
        poisoned_clients.map(poison_model)
        assert all(
            poisoned_clients._models[i]["model"].get_params()["n_neighbors"] == 6
            for i in poisoned_clients._models
        )

        with pytest.raises(AssertionError):
            poisoned_clients.map(bad_poison_model)
