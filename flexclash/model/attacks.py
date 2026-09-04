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
from typing import Any, Callable, List, Optional, Sequence, Union

import tensorly as tl
from flex.model import FlexModel
from flex.pool.aggregators import fed_avg_f, set_tensorly_backend

from flexclash.model.poisoning_decorators import model_poisoner


def extract_model_weights(model_or_container: Any) -> List[Any]:
    """
    Extracts a list of layer weights from a FlexModel, PyTorch nn.Module,
    TensorFlow Model, or list of weights.

    Args:
    -----
        model_or_container: Object containing model parameters (FlexModel,
            torch.nn.Module, tf.keras.Model, or list of tensors/arrays).

    Returns:
    --------
        List[Any]: List of layer tensors or arrays.
    """
    if isinstance(model_or_container, FlexModel):
        if "model" in model_or_container and model_or_container["model"] is not None:
            return extract_model_weights(model_or_container["model"])
        elif "weights" in model_or_container and model_or_container["weights"] is not None:
            return extract_model_weights(model_or_container["weights"])
        else:
            raise ValueError("FlexModel does not contain 'model' or 'weights'.")

    # PyTorch nn.Module
    if hasattr(model_or_container, "state_dict") and callable(
        getattr(model_or_container, "state_dict")
    ):
        import torch

        weights = []
        for name, tensor in model_or_container.state_dict().items():
            if "num_batches_tracked" in name or (
                hasattr(tensor, "is_floating_point") and not tensor.is_floating_point()
            ):
                weights.append(torch.tensor([]))
            else:
                weights.append(tensor.detach().clone())
        return weights

    # TensorFlow / Keras Model
    if hasattr(model_or_container, "get_weights") and callable(
        getattr(model_or_container, "get_weights")
    ):
        return [w.copy() for w in model_or_container.get_weights()]

    # List or tuple of weights
    if isinstance(model_or_container, (list, tuple)):
        return list(model_or_container)

    raise TypeError(
        f"Unsupported model type for weight extraction: {type(model_or_container)}"
    )


def set_model_weights(model_or_container: Any, new_weights: Sequence[Any]) -> None:
    """
    Sets a list of layer weights into a FlexModel, PyTorch nn.Module,
    TensorFlow Model, or weight container.

    Args:
    -----
        model_or_container: Object to update (FlexModel, torch.nn.Module,
            tf.keras.Model, or list).
        new_weights: New weights to apply.
    """
    if isinstance(model_or_container, FlexModel):
        if "model" in model_or_container and model_or_container["model"] is not None:
            m = model_or_container["model"]
            if hasattr(m, "state_dict") or hasattr(m, "set_weights"):
                set_model_weights(m, new_weights)
            elif isinstance(m, list):
                model_or_container["model"] = list(new_weights)
            else:
                set_model_weights(m, new_weights)
        elif "weights" in model_or_container:
            model_or_container["weights"] = list(new_weights)
        else:
            model_or_container["weights"] = list(new_weights)
        return

    # PyTorch nn.Module
    if hasattr(model_or_container, "state_dict") and callable(
        getattr(model_or_container, "state_dict")
    ):
        import torch

        with torch.no_grad():
            for (name, param), new_w in zip(
                model_or_container.state_dict().items(), new_weights
            ):
                if hasattr(new_w, "numel") and new_w.numel() == 0:
                    continue
                try:
                    if len(new_w) == 0:
                        continue
                except (TypeError, IndexError):
                    pass
                if hasattr(param, "is_floating_point") and not param.is_floating_point():
                    continue
                param.copy_(new_w)
        return

    # TensorFlow / Keras Model
    if hasattr(model_or_container, "set_weights") and callable(
        getattr(model_or_container, "set_weights")
    ):
        import numpy as np

        model_or_container.set_weights([np.asarray(w) for w in new_weights])
        return

    # List container
    if isinstance(model_or_container, list):
        model_or_container.clear()
        model_or_container.extend(new_weights)
        return

    raise TypeError(
        f"Unsupported model type for setting weights: {type(model_or_container)}"
    )


def extract_pool_weights(pool_or_clients: Any) -> List[List[Any]]:
    """
    Extracts lists of layer weights from each client in a FlexPool or sequence of models.

    Args:
    -----
        pool_or_clients: FlexPool, list of FlexModel, or dict of models.

    Returns:
    --------
        List[List[Any]]: List containing each client's layer weights.
    """
    if hasattr(pool_or_clients, "_models"):
        models = pool_or_clients._models.values()
    elif isinstance(pool_or_clients, (list, tuple)):
        models = pool_or_clients
    elif isinstance(pool_or_clients, dict):
        models = pool_or_clients.values()
    else:
        raise TypeError(
            f"Cannot extract weights from pool of type: {type(pool_or_clients)}"
        )

    weights_list = [extract_model_weights(m) for m in models]
    if not weights_list:
        raise ValueError("The provided client pool has no clients/models.")
    return weights_list


def inner_product_manipulation_f(
    honest_weights: List[List[Any]],
    server_weights: List[Any],
    epsilon: float = 1.0,
) -> List[Any]:
    """
    Low-level functional implementation of Inner Product Manipulation (IPM).

    Reference:
        Xie et al. (UAI 2020), "Fall of Empires: Breaking Byzantine-tolerant SGD
        by Inner Product Manipulation", PMLR 115:399-408.
        https://proceedings.mlr.press/v115/xie20a.html

    Given honest client weights w_i for i in H and global server weights w_server:
        - Benign update: Δw_i = w_i - w_server
        - Mean benign update: Δw_bar = (1 / |H|) * sum_{i in H} Δw_i
        - Malicious update: v = -epsilon * Δw_bar
        - Malicious weights: w_mal = w_server + v = w_server - epsilon * Δw_bar

    Args:
    -----
        honest_weights: List of weights from honest clients. Each element is
            a list of tensors/arrays corresponding to model layers.
        server_weights: List of tensors/arrays of the server model layers.
        epsilon: Attack strength factor (positive float). Defaults to 1.0.

    Returns:
    --------
        List[Any]: Poisoned model weights (list of layer tensors/arrays).
    """
    if not honest_weights:
        raise ValueError("honest_weights must contain at least one client's weights.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    # Check for empty placeholder layers (e.g. ignored non-floating parameters)
    active_indices = []
    for i, w_srv in enumerate(server_weights):
        is_empty = False
        if hasattr(w_srv, "numel") and w_srv.numel() == 0:
            is_empty = True
        elif hasattr(w_srv, "size") and getattr(w_srv, "size", 1) == 0:
            is_empty = True
        elif isinstance(w_srv, (list, tuple)) and len(w_srv) == 0:
            is_empty = True
        if not is_empty:
            active_indices.append(i)

    if not active_indices:
        return list(server_weights)

    active_honest_weights = [
        [hw[idx] for idx in active_indices] for hw in honest_weights
    ]
    active_server_weights = [server_weights[idx] for idx in active_indices]

    set_tensorly_backend(active_honest_weights + [active_server_weights])
    mean_honest_weights = fed_avg_f(active_honest_weights)

    malicious_weights = list(server_weights)
    for idx, w_srv, w_h_mean in zip(
        active_indices, active_server_weights, mean_honest_weights
    ):
        context = tl.context(w_srv)
        diff = w_h_mean - w_srv
        eps_tensor = tl.tensor(epsilon, **context)
        w_mal = w_srv - eps_tensor * diff
        malicious_weights[idx] = w_mal

    return malicious_weights


def ipm_poisoner(
    server_model: Any,
    honest_clients: Any,
    epsilon: float = 1.0,
) -> Callable:
    """
    Creates a client model poisoner for Inner Product Manipulation (IPM).

    Precomputes the poisoned weights from the server_model and honest_clients,
    and returns a function decorated with @model_poisoner suitable for
    `malicious_pool.map(poison_fn)`.

    Args:
    -----
        server_model: The server / global model (FlexModel, nn.Module, keras.Model,
            or list of weights).
        honest_clients: The honest clients (FlexPool, list of FlexModel, or list of weights)
            used to compute the benign update direction.
        epsilon: Attack strength factor (positive float). Defaults to 1.0.

    Returns:
    --------
        Callable: A function decorated with @model_poisoner.
    """
    server_weights = extract_model_weights(server_model)
    if isinstance(honest_clients, list) and honest_clients and isinstance(honest_clients[0], list):
        honest_weights = honest_clients
    else:
        honest_weights = extract_pool_weights(honest_clients)

    poisoned_weights = inner_product_manipulation_f(
        honest_weights, server_weights, epsilon=epsilon
    )

    @model_poisoner
    def _ipm_poison_client(client_model: FlexModel):
        set_model_weights(client_model, poisoned_weights)
        return client_model

    return _ipm_poison_client


def inner_product_manipulation(
    malicious_pool_or_clients: Any,
    server_model: Any,
    honest_pool: Optional[Any] = None,
    malicious_clients: Optional[Union[int, Sequence[Any], Any]] = None,
    epsilon: float = 1.0,
) -> Any:
    """
    Executes the Inner Product Manipulation (IPM) model poisoning attack on malicious clients.

    Supports two calling patterns:
    1. Explicit pools:
       `inner_product_manipulation(malicious_pool, server_model, honest_pool=honest_pool, epsilon=1.0)`
    2. Unified client pool with malicious specification:
       `inner_product_manipulation(selected_pool, server_model, malicious_clients=[0, 1], epsilon=1.0)`

    Args:
    -----
        malicious_pool_or_clients: Either the pool of malicious clients (when honest_pool
            is provided) or the entire client pool (when malicious_clients is provided).
        server_model: Global server model before the current training round.
        honest_pool: Pool or list of honest clients (optional if malicious_clients is provided).
        malicious_clients: Indices, actor IDs, count, or sub-pool designating malicious clients
            within malicious_pool_or_clients.
        epsilon: Attack strength hyperparameter. Defaults to 1.0.

    Returns:
    --------
        The malicious clients pool that was poisoned.
    """
    if honest_pool is not None:
        mal_pool = malicious_pool_or_clients
        hon_pool = honest_pool
    elif malicious_clients is not None:
        client_pool = malicious_pool_or_clients
        if not hasattr(client_pool, "select"):
            raise TypeError(
                "malicious_pool_or_clients must be a FlexPool when using malicious_clients."
            )

        if isinstance(malicious_clients, int):
            mal_pool = client_pool.select(malicious_clients)
            hon_pool = client_pool.select(
                lambda aid, _: aid not in mal_pool.actor_ids
            )
        elif hasattr(malicious_clients, "actor_ids"):
            mal_pool = malicious_clients
            hon_pool = client_pool.select(
                lambda aid, _: aid not in mal_pool.actor_ids
            )
        elif isinstance(malicious_clients, (list, tuple, set)):
            mal_set = set(malicious_clients)
            mal_pool = client_pool.select(lambda aid, _: aid in mal_set)
            hon_pool = client_pool.select(lambda aid, _: aid not in mal_set)
        else:
            raise TypeError(
                f"Unsupported malicious_clients type: {type(malicious_clients)}"
            )
    else:
        raise ValueError(
            "Either honest_pool or malicious_clients must be provided to perform Inner Product Manipulation."
        )

    poison_fn = ipm_poisoner(server_model, hon_pool, epsilon=epsilon)

    if hasattr(mal_pool, "map"):
        mal_pool.map(poison_fn)
    elif isinstance(mal_pool, (list, tuple)):
        for client in mal_pool:
            poison_fn(client, None)
    else:
        raise TypeError(f"Unsupported malicious pool type: {type(mal_pool)}")

    return mal_pool
