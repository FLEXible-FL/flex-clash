import random
import unittest

import pytest
import tensorly as tl

from flexclash.pool import bulyan
from flexclash.pool import central_differential_privacy as cdp
from flexclash.pool import median, multikrum, trimmed_mean


def simulate_clients_weights_for_module(n_clients, modulename):
    framework = __import__(modulename)
    n_clients = 5
    simulated_client_weights = []
    num_layers = 5
    num_dim = [1, 2, 3, 4, 5]
    layer_ndims = random.sample(num_dim, k=num_layers)
    # n-1 clients with ones
    for _ in range(n_clients - 1):
        simulated_weights = []
        random.seed(0)
        for ndims in layer_ndims:
            tmp_dims = random.sample(
                num_dim, k=ndims
            )  # ndims dimensions of num_dim sizes
            simulated_weights.append(framework.ones(tmp_dims))
        simulated_client_weights.append(simulated_weights)
    # Another client with only zeros
    simulated_bad_weights = []
    random.seed(0)
    for ndims in layer_ndims:
        tmp_dims = random.sample(num_dim, k=ndims)  # ndims dimensions of num_dim sizes
        simulated_bad_weights.append(framework.zeros(tmp_dims))
    simulated_client_weights.append(simulated_bad_weights)
    return {"weights": simulated_client_weights}


class TestFlexAggregators(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_weights(self):
        self._torch_weights = simulate_clients_weights_for_module(
            n_clients=5, modulename="torch"
        )
        self._tf_weights = simulate_clients_weights_for_module(
            n_clients=5, modulename="tensorflow"
        )
        self._np_weights = simulate_clients_weights_for_module(
            n_clients=5, modulename="numpy"
        )

    def test_fed_median_with_torch(self):
        median(self._torch_weights, None)
        agg_weights = self._torch_weights["aggregated_weights"]
        assert tl.get_backend() == "pytorch"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_median_with_tf(self):
        median(self._tf_weights, None)
        agg_weights = self._tf_weights["aggregated_weights"]
        assert tl.get_backend() == "tensorflow"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_median_with_np(self):
        median(self._np_weights, None)
        agg_weights = self._np_weights["aggregated_weights"]
        assert tl.get_backend() == "numpy"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_trimmed_mean_with_torch(self):
        trimmed_mean(self._torch_weights, None, trim_proportion=0.2)
        agg_weights = self._torch_weights["aggregated_weights"]
        assert tl.get_backend() == "pytorch"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_trimmed_mean_with_tf(self):
        trimmed_mean(self._tf_weights, None, trim_proportion=0.2)
        agg_weights = self._tf_weights["aggregated_weights"]
        assert tl.get_backend() == "tensorflow"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_trimmed_mean_with_np(self):
        trimmed_mean(self._np_weights, None, trim_proportion=0.2)
        agg_weights = self._np_weights["aggregated_weights"]
        assert tl.get_backend() == "numpy"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_multikrum_with_torch(self):
        multikrum(self._torch_weights, None)
        agg_weights = self._torch_weights["aggregated_weights"]
        assert tl.get_backend() == "pytorch"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_multikrum_with_tf(self):
        multikrum(self._tf_weights, None)
        agg_weights = self._tf_weights["aggregated_weights"]
        assert tl.get_backend() == "tensorflow"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_multikrum_with_np(self):
        multikrum(self._np_weights, None)
        agg_weights = self._np_weights["aggregated_weights"]
        assert tl.get_backend() == "numpy"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_bulyan_with_torch(self):
        bulyan(self._torch_weights, None, m=1)
        agg_weights = self._torch_weights["aggregated_weights"]
        assert tl.get_backend() == "pytorch"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_bulyan_with_tf(self):
        bulyan(self._tf_weights, None, m=1)
        agg_weights = self._tf_weights["aggregated_weights"]
        assert tl.get_backend() == "tensorflow"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_bulyan_with_np(self):
        bulyan(self._np_weights, None, m=1)
        agg_weights = self._np_weights["aggregated_weights"]
        assert tl.get_backend() == "numpy"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_cdp_with_torch(self):
        cdp(self._torch_weights, None, noise_multiplier=0, l2_clip=9999)
        agg_weights = self._torch_weights["aggregated_weights"]
        assert tl.get_backend() == "pytorch"
        assert all(
            tl.all(w == tl.tensor(0.8, **tl.context(w)) * tl.ones(tl.shape(w)))
            for w in agg_weights
        )

    def test_fed_cdp_with_tf(self):
        cdp(self._tf_weights, None, noise_multiplier=0, l2_clip=9999)
        agg_weights = self._tf_weights["aggregated_weights"]
        assert tl.get_backend() == "tensorflow"
        assert all(
            tl.all(
                w
                == tl.tensor(0.8, **tl.context(w))
                * tl.ones(tl.shape(w), **tl.context(w))
            )
            for w in agg_weights
        )

    def test_fed_cdp_with_np(self):
        cdp(self._np_weights, None, noise_multiplier=0, l2_clip=9999)
        agg_weights = self._np_weights["aggregated_weights"]
        assert tl.get_backend() == "numpy"
        assert all(
            tl.all(
                w
                == tl.tensor(0.8, **tl.context(w))
                * tl.ones(tl.shape(w), **tl.context(w))
            )
            for w in agg_weights
        )
