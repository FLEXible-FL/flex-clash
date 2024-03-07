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
"""File that contains the adapted server defences based on aggregators in FLEXible.

For the moment, we include:
- Median
- Trimmed mean
- Krum
- Bulyan
"""

from math import sqrt

import tensorly as tl
from flex.pool.aggregators import fed_avg_f, set_tensorly_backend
from flex.pool.decorators import aggregate_weights
from numpy.random import normal


def generalized_percentile_aggregator_f(
    list_of_weights: list, percentile: [slice, int]
):
    agg_weights = []
    number_of_layers = len(list_of_weights[0])
    for layer_index in range(number_of_layers):
        weights_per_layer = [weights[layer_index] for weights in list_of_weights]
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.sort(weights_per_layer, axis=0)[percentile]
        agg_weights.append(agg_layer)
    return agg_weights


def median_f(list_of_weights: list):
    num_clients = len(list_of_weights)
    median_pos = num_clients // 2
    return generalized_percentile_aggregator_f(list_of_weights, median_pos)


def trimmed_mean_f(list_of_weights: list, trim_proportion):
    num_clients = len(list_of_weights)
    min_trim = round(trim_proportion * num_clients)
    max_trim = round((1 - trim_proportion) * num_clients)
    return generalized_percentile_aggregator_f(
        list_of_weights, slice(min_trim, max_trim + 1)
    )


def compute_distance_matrix(list_of_weights: list):
    num_clients = len(list_of_weights)
    distance_matrix = [list(range(num_clients)) for i in range(num_clients)]
    for i in range(num_clients):
        w_i = list_of_weights[i]
        for j in range(i, num_clients):
            w_j = list_of_weights[j]
            tmp_dist = sum([tl.norm(a - b) ** 2 for a, b in zip(w_i, w_j)])
            distance_matrix[i][j] = tmp_dist
            distance_matrix[j][i] = tmp_dist
    return distance_matrix


def central_differential_privacy_f(list_of_weights: list, l2_clip, noise_multiplier):
    num_clients = len(list_of_weights)
    for i in range(num_clients):
        tmp_dist = sum([tl.norm(w) ** 2 for w in list_of_weights[i]])
        l2_norm = sqrt(tmp_dist) + 1e-12
        clip_ratio = min(1, l2_clip / l2_norm)
        for j, w in enumerate(list_of_weights[i]):
            context = tl.context(w)
            clip_ratio = tl.tensor(clip_ratio, **context)
            list_of_weights[i][j] = w * clip_ratio
    agg_weights = fed_avg_f(list_of_weights)
    noise_ratio = l2_clip * noise_multiplier / num_clients
    for i, w in enumerate(agg_weights):
        context = tl.context(agg_weights[i])
        noise = tl.tensor(
            normal(loc=0.0, scale=noise_ratio, size=tl.shape(w)), **context
        )
        agg_weights[i] = w + noise
    return agg_weights


@aggregate_weights
def central_differential_privacy(
    list_of_weights: list, l2_clip=1, noise_multiplier=0.1
):
    set_tensorly_backend(list_of_weights)
    return central_differential_privacy_f(list_of_weights, l2_clip, noise_multiplier)


@aggregate_weights
def median(list_of_weights: list):
    set_tensorly_backend(list_of_weights)
    return median_f(list_of_weights)


@aggregate_weights
def trimmed_mean(list_of_weights, trim_proportion=0.1):
    set_tensorly_backend(list_of_weights)
    return trimmed_mean_f(list_of_weights, trim_proportion)


def krum_criteria(distance_matrix, f, m):
    num_clients = len(distance_matrix)
    # Compute scores
    scores = []
    num_selected = num_clients - f - 2
    for i in range(num_clients):
        completed_scores = distance_matrix[i]
        completed_scores.sort()
        scores.append(
            sum(completed_scores[1 : num_selected + 1])
        )  # distance to oneself is always first
    # We associate each client with her scores and sort them using her scores
    pairs = [(i, scores[i]) for i in range(num_clients)]
    pairs.sort(key=lambda pair: pair[1])
    return pairs[:m]


@aggregate_weights
def multikrum(list_of_weights: list, f=1, m=5):
    set_tensorly_backend(list_of_weights)
    distance_matrix = compute_distance_matrix(list_of_weights)
    pairs = krum_criteria(distance_matrix, f, m)
    selected_weights = [list_of_weights[i] for i, _ in pairs]
    return median_f(selected_weights)


@aggregate_weights
def bulyan(list_of_weights: list, f=1, m=5):
    set_tensorly_backend(list_of_weights)
    num_clients = len(list_of_weights)
    distance_matrix = compute_distance_matrix(list_of_weights)
    # Using the krum criteria, select each time a client
    selected_clients = []
    max_selected_clients = num_clients - 2 * m
    while len(selected_clients) < max_selected_clients:
        pairs = krum_criteria(distance_matrix, f, m)
        selected_client_index = pairs[0][0]
        selected_clients.append(selected_client_index)
        # delete selected client distances, by setting her distances to infinity
        for i, _ in enumerate(distance_matrix[selected_client_index]):
            distance_matrix[selected_client_index][i] = float("inf")
            distance_matrix[i][selected_client_index] = float("inf")

    selected_weights = [list_of_weights[i] for i in selected_clients]

    return trimmed_mean_f(selected_weights, m / len(selected_clients))
