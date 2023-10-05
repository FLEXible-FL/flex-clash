"""File that contains the adapted server defences based on aggregators in FLEXible.

For the moment, we include:
- Median
- Trimmed mean
- Krum
- Bulyan
"""

import tensorly as tl
from flex.pool.aggregators import set_tensorly_backend
from flex.pool.decorators import aggregate_weights


def generalized_percentile_aggregator_f(list_of_weights: list, percentile: [slice, int]):
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


def trimmed_mean_f(list_of_weights: list, trim_proportion=0.1):
    num_clients = len(list_of_weights)
    min_trim = round(trim_proportion * num_clients)
    max_trim = round((1 - trim_proportion) * num_clients)
    return generalized_percentile_aggregator_f(list_of_weights, slice(min_trim, max_trim+1))


def compute_distance_matrix(list_of_weights: list):
    num_clients = len(list_of_weights)
    distance_matrix = [list(range(num_clients)) for i in range(num_clients)]
    for i in range(num_clients):
        w_i = list_of_weights[i]
        tmp_dist = 0
        for j in range(i, num_clients):
            w_j = list_of_weights[j]
            tmp_dist += sum([tl.norm(tl.tensor(a - b)) ** 2 for a, b in zip(w_i, w_j)])
            distance_matrix[i][j] = tmp_dist
            distance_matrix[j][i] = tmp_dist
    return distance_matrix


@aggregate_weights
def median(list_of_weights: list, *args, **kwargs):
    set_tensorly_backend(list_of_weights)
    return median_f(list_of_weights, *args, **kwargs)


@aggregate_weights
def trimmed_mean(list_of_weights, *args, **kwargs):
    set_tensorly_backend(list_of_weights)
    return trimmed_mean_f(list_of_weights, *args, **kwargs)


@aggregate_weights
def multikrum(list_of_weights: list, f=1, m=5):
    set_tensorly_backend(list_of_weights)
    num_clients = len(list_of_weights)
    # Compute matrix of distances
    distance_matrix = compute_distance_matrix(list_of_weights)
    # Compute scores
    scores = []
    num_selected = num_clients - f - 2
    for i in range(num_clients):
        completed_scores = distance_matrix[i]
        completed_scores.sort()
        scores.append(sum(completed_scores[1:num_selected+1])) # distance to oneself is always first
    # We associate each client with her scores and sort them using her scores
    pairs = [(i, scores[i]) for i in range(num_clients)]
    pairs.sort(key=lambda pair: pair[1])
    selected_weights = [list_of_weights[i] for i, _ in pairs[:m]]
    return median_f(selected_weights)


@aggregate_weights
def bulyan(list_of_weights: list, f=1, m=5):
    set_tensorly_backend(list_of_weights)
    num_clients = len(list_of_weights)
    # Compute matrix of distances
    distance_matrix = compute_distance_matrix(list_of_weights)
    # Using the krum criteria, select each time a client
    selected_clients = []
    krum_num_selected = num_clients - f - 2
    max_selected_clients = num_clients - 2 * m
    while len(selected_clients) < max_selected_clients:
        scores = list(range(num_clients))
        available_clients = list(range(num_clients))
        available_clients = list(set(available_clients) - set(selected_clients))
        for i in available_clients:
            completed_scores = distance_matrix[i]
            completed_scores.sort()
            scores[i] = sum(completed_scores[1:krum_num_selected+1]) # distance to oneself is always first
        # We associate each client with her scores and sort them using her scores
        pairs = [(i, scores[i]) for i in available_clients]
        pairs.sort(key=lambda pair: pair[1])
        # Get selected clients params
        selected_client_index = pairs[0][0]
        selected_clients.append(selected_client_index)
        # delete selected client distances
        for i, row in enumerate(distance_matrix):
            row[selected_client_index] = 0

    selected_weights = [list_of_weights[i] for i in selected_clients]

    return trimmed_mean_f(selected_weights, m/len(selected_clients))
