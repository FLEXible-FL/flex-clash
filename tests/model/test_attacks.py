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

import numpy as np
import torch
import torch.nn as nn
from flex.data import Dataset, FedDataDistribution
from flex.model import FlexModel
from flex.pool import FlexPool, deploy_server_model, init_server_model

from flexclash.model import (
    inner_product_manipulation,
    inner_product_manipulation_f,
    ipm_poisoner,
)
from flexclash.model.attacks import (
    extract_model_weights,
    set_model_weights,
)


class TestInnerProductManipulation(unittest.TestCase):
    def test_ipm_f_math_numpy(self):
        # 3 honest clients, 2 layers
        w_h1 = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, -0.5])]
        w_h2 = [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([1.5, 0.5])]
        w_h3 = [np.array([[3.0, 4.0], [5.0, 6.0]]), np.array([1.0, 0.0])]
        honest_weights = [w_h1, w_h2, w_h3]

        # Server model
        w_srv = [np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([0.0, 0.0])]

        # Mean honest weights
        # layer 0: [[2.0, 3.0], [4.0, 5.0]]
        # layer 1: [1.0, 0.0]
        # Mean delta:
        # layer 0: [[1.0, 2.0], [3.0, 4.0]]
        # layer 1: [1.0, 0.0]

        for eps in [0.5, 1.0, 2.0]:
            w_mal = inner_product_manipulation_f(
                honest_weights, w_srv, epsilon=eps
            )
            # w_mal should equal w_srv - eps * (w_h_mean - w_srv)
            expected_l0 = w_srv[0] - eps * (
                np.array([[2.0, 3.0], [4.0, 5.0]]) - w_srv[0]
            )
            expected_l1 = w_srv[1] - eps * (np.array([1.0, 0.0]) - w_srv[1])
            np.testing.assert_allclose(w_mal[0], expected_l0, rtol=1e-5)
            np.testing.assert_allclose(w_mal[1], expected_l1, rtol=1e-5)

            # Test inner product property: <v, delta_bar> = -eps * ||delta_bar||^2 < 0
            delta_bar = [
                np.array([[2.0, 3.0], [4.0, 5.0]]) - w_srv[0],
                np.array([1.0, 0.0]) - w_srv[1],
            ]
            v = [w_mal[0] - w_srv[0], w_mal[1] - w_srv[1]]
            ip = sum(np.sum(vi * dbi) for vi, dbi in zip(v, delta_bar))
            expected_norm_sq = sum(np.sum(dbi ** 2) for dbi in delta_bar)
            np.testing.assert_allclose(ip, -eps * expected_norm_sq, rtol=1e-5)
            self.assertLess(ip, 0)

    def test_ipm_f_torch(self):
        w_h1 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        w_h2 = [torch.tensor([3.0, 4.0]), torch.tensor([5.0])]
        w_srv = [torch.tensor([0.0, 0.0]), torch.tensor([0.0])]

        w_mal = inner_product_manipulation_f(
            [w_h1, w_h2], w_srv, epsilon=1.0
        )

        # Mean honest: [2.0, 3.0], [4.0]
        # Malicious: [0.0 - 2.0, 0.0 - 3.0], [0.0 - 4.0] = [-2.0, -3.0], [-4.0]
        self.assertTrue(torch.allclose(w_mal[0], torch.tensor([-2.0, -3.0])))
        self.assertTrue(torch.allclose(w_mal[1], torch.tensor([-4.0])))

    def test_ipm_f_empty_layers(self):
        # Empty placeholder layer in the middle
        w_h1 = [torch.tensor([1.0, 2.0]), torch.tensor([]), torch.tensor([3.0])]
        w_h2 = [torch.tensor([3.0, 4.0]), torch.tensor([]), torch.tensor([5.0])]
        w_srv = [torch.tensor([0.0, 0.0]), torch.tensor([]), torch.tensor([0.0])]

        w_mal = inner_product_manipulation_f(
            [w_h1, w_h2], w_srv, epsilon=1.0
        )
        self.assertEqual(len(w_mal), 3)
        self.assertEqual(w_mal[1].numel(), 0)
        self.assertTrue(torch.allclose(w_mal[0], torch.tensor([-2.0, -3.0])))
        self.assertTrue(torch.allclose(w_mal[2], torch.tensor([-4.0])))

    def test_ipm_f_invalid_inputs(self):
        w_srv = [np.array([1.0])]
        with self.assertRaises(ValueError):
            inner_product_manipulation_f([], w_srv, epsilon=1.0)
        with self.assertRaises(ValueError):
            inner_product_manipulation_f([[np.array([2.0])]], w_srv, epsilon=0.0)
        with self.assertRaises(ValueError):
            inner_product_manipulation_f([[np.array([2.0])]], w_srv, epsilon=-1.0)

    def test_extract_and_set_model_weights_torch(self):
        net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        weights = extract_model_weights(net)
        self.assertEqual(len(weights), 4)  # 2 weights + 2 biases

        new_weights = [torch.ones_like(w) * 5.0 for w in weights]
        set_model_weights(net, new_weights)

        for p in net.parameters():
            self.assertTrue(torch.allclose(p, torch.tensor(5.0)))

    def test_ipm_with_flex_pool_explicit(self):
        # Build server model with PyTorch
        @init_server_model
        def build_server_model():
            fm = FlexModel()
            fm["model"] = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                fm["model"].weight.copy_(torch.tensor([[1.0, 1.0]]))
            return fm

        @deploy_server_model
        def copy_to_clients(server_flex_model: FlexModel):
            return copy.deepcopy(server_flex_model)

        dummy_data = Dataset.from_array(np.zeros((10, 2)), np.zeros(10))
        fed_data = FedDataDistribution.iid_distribution(dummy_data, n_nodes=4)

        pool = FlexPool.client_server_pool(fed_data, init_func=build_server_model)
        server = pool.servers
        clients = pool.clients
        server.map(copy_to_clients, clients)

        # Let honest clients be client 0 and 1, simulate updates
        actor_ids = list(clients.actor_ids)
        hon_ids = actor_ids[:2]
        mal_ids = actor_ids[2:]

        hon_pool = clients.select(lambda aid, _: aid in hon_ids)
        mal_pool = clients.select(lambda aid, _: aid in mal_ids)

        # Update honest clients
        with torch.no_grad():
            hon_pool._models[hon_ids[0]]["model"].weight.copy_(
                torch.tensor([[2.0, 3.0]])
            )
            hon_pool._models[hon_ids[1]]["model"].weight.copy_(
                torch.tensor([[4.0, 5.0]])
            )

        # Average honest weight: [[3.0, 4.0]]
        # Global server weight: [[1.0, 1.0]]
        # Delta bar: [[2.0, 3.0]]
        # Malicious weight: [[1.0 - 2.0, 1.0 - 3.0]] = [[-1.0, -2.0]]

        inner_product_manipulation(
            mal_pool,
            server_model=server._models[list(server.actor_ids)[0]],
            honest_pool=hon_pool,
            epsilon=1.0,
        )

        expected_mal_weight = torch.tensor([[-1.0, -2.0]])
        for aid in mal_ids:
            actual = mal_pool._models[aid]["model"].weight
            self.assertTrue(
                torch.allclose(actual, expected_mal_weight),
                f"Client {aid} weight mismatch: {actual} vs {expected_mal_weight}",
            )

    def test_ipm_with_flex_pool_unified(self):
        @init_server_model
        def build_server_model():
            fm = FlexModel()
            fm["model"] = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                fm["model"].weight.copy_(torch.tensor([[0.0, 0.0]]))
            return fm

        @deploy_server_model
        def copy_to_clients(server_flex_model: FlexModel):
            return copy.deepcopy(server_flex_model)

        dummy_data = Dataset.from_array(np.zeros((10, 2)), np.zeros(10))
        fed_data = FedDataDistribution.iid_distribution(dummy_data, n_nodes=5)

        pool = FlexPool.client_server_pool(fed_data, init_func=build_server_model)
        server = pool.servers
        clients = pool.clients
        server.map(copy_to_clients, clients)

        actor_ids = list(clients.actor_ids)
        mal_ids = actor_ids[:2]
        hon_ids = actor_ids[2:]

        # Set honest clients to [[1.0, 1.0]]
        with torch.no_grad():
            for hid in hon_ids:
                clients._models[hid]["model"].weight.copy_(torch.tensor([[1.0, 1.0]]))

        # Execute IPM on unified client pool
        inner_product_manipulation(
            clients,
            server_model=server._models[list(server.actor_ids)[0]],
            malicious_clients=mal_ids,
            epsilon=1.0,
        )

        expected_mal = torch.tensor([[-1.0, -1.0]])
        for mid in mal_ids:
            self.assertTrue(
                torch.allclose(clients._models[mid]["model"].weight, expected_mal)
            )

        # Ensure honest clients were NOT modified
        expected_hon = torch.tensor([[1.0, 1.0]])
        for hid in hon_ids:
            self.assertTrue(
                torch.allclose(clients._models[hid]["model"].weight, expected_hon)
            )

    def test_ipm_poisoner_map(self):
        @init_server_model
        def build_server_model():
            fm = FlexModel()
            fm["model"] = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                fm["model"].weight.copy_(torch.tensor([[0.0, 0.0]]))
            return fm

        @deploy_server_model
        def copy_to_clients(server_flex_model: FlexModel):
            return copy.deepcopy(server_flex_model)

        dummy_data = Dataset.from_array(np.zeros((10, 2)), np.zeros(10))
        fed_data = FedDataDistribution.iid_distribution(dummy_data, n_nodes=3)

        pool = FlexPool.client_server_pool(fed_data, init_func=build_server_model)
        server = pool.servers
        clients = pool.clients
        server.map(copy_to_clients, clients)

        actor_ids = list(clients.actor_ids)
        hon_pool = clients.select(lambda aid, _: aid == actor_ids[0])
        mal_pool = clients.select(lambda aid, _: aid in actor_ids[1:])

        with torch.no_grad():
            hon_pool._models[actor_ids[0]]["model"].weight.copy_(
                torch.tensor([[2.0, 4.0]])
            )

        srv_model = server._models[list(server.actor_ids)[0]]
        poison_fn = ipm_poisoner(srv_model, hon_pool, epsilon=0.5)

        mal_pool.map(poison_fn)

        # delta_bar = [[2.0, 4.0]]
        # w_mal = 0 - 0.5 * [[2.0, 4.0]] = [[-1.0, -2.0]]
        expected_mal = torch.tensor([[-1.0, -2.0]])
        for aid in actor_ids[1:]:
            self.assertTrue(
                torch.allclose(mal_pool._models[aid]["model"].weight, expected_mal)
            )
