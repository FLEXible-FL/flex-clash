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


@aggregate_weights
def median(list_of_weights: list, *args, **kwargs):
    set_tensorly_backend(list_of_weights)
    return median_f(list_of_weights, *args, **kwargs)


def median_f(list_of_weights: list):
    agg_weights = []
    number_of_layers = len(list_of_weights[0])
    for layer_index in range(number_of_layers):
        weights_per_layer = [weights[layer_index] for weights in list_of_weights]
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.sort(weights_per_layer, axis=0)[len(weights_per_layer) // 2]
        agg_weights.append(agg_layer)
    return agg_weights


@aggregate_weights
def trimmed_mean(list_of_weights, *args, **kwargs):
    set_tensorly_backend(list_of_weights)
    return trimmed_mean_f(list_of_weights, *args, **kwargs)


def trimmed_mean_f(list_of_weights: list, trim_proportion=0.1):
    agg_weights = []
    number_of_layers = len(list_of_weights[0])
    for layer_index in range(number_of_layers):
        weights_per_layer = [weights[layer_index] for weights in list_of_weights]
        weights_per_layer = tl.stack(weights_per_layer)
        min_trim = round(trim_proportion * len(weights_per_layer))
        max_trim = round((1 - trim_proportion) * len(weights_per_layer))
        weights_per_layer = tl.sort(weights_per_layer, axis=0)[min_trim:max_trim]
        agg_layer = tl.mean(weights_per_layer, axis=0)
        agg_weights.append(agg_layer)
    return agg_weights


@aggregate_weights
def multikrum(list_of_weights: list, f=1, m=5):
    set_tensorly_backend(list_of_weights)
    num_clients = len(list_of_weights)

    # Calculate matrix of distances
    distance_matrix = [list() for i in range(num_clients)]
    for i in range(num_clients):
        w_i = list_of_weights[i]
        tmp_dist = 0
        for j in range(i + 1, num_clients):
            w_j = list_of_weights[j]
            tmp_dist += sum([tl.norm(tl.tensor(a - b)) ** 2 for a, b in zip(w_i, w_j)])
            distance_matrix.append(tmp_dist)
            # Simetric matrix
            distance_matrix[j].append(tmp_dist)

    scores = [list() for i in range(num_clients)]
    num_selected = num_clients - f - 2
    for i in range(num_clients):
        completed_scores = distance_matrix[i]
        completed_scores.sort()
        scores[i] = sum(completed_scores[:num_selected])

    # We associate the clients params with the scores associated and sort it using the scores
    pairs = [(i, scores[i]) for i in range(num_clients)]
    pairs.sort(key=lambda pair: pair[1])

    selected_weights = [list_of_weights[i] for i, _ in pairs[:m]]

    return median_f(selected_weights)


@aggregate_weights
def bulyan(list_of_weights: list, f=1, m=5):
    set_tensorly_backend(list_of_weights)
    num_clients = len(list_of_weights)

    # Calculate matrix of distances
    distance_matrix = [list() for i in range(num_clients)]
    for i in range(num_clients):
        w_i = list_of_weights[i]
        tmp_dist = 0
        for j in range(i + 1, num_clients):
            w_j = list_of_weights[j]
            tmp_dist += sum([tl.norm(tl.tensor(a - b)) ** 2 for a, b in zip(w_i, w_j)])
            distance_matrix.append(tmp_dist)
            # Simetric matrix
            distance_matrix[j].append(tmp_dist)

    # Calculate score matrix
    selected_clients = []
    while len(selected_clients) < num_clients - 2 * m:
        scores = [list() for i in range(num_clients)]
        num_selected = num_clients - f - 2
        for i in range(num_clients):
            completed_scores = distance_matrix[i]
            completed_scores.sort()
            scores[i] = sum(completed_scores[:num_selected])

    # We associate the clients params with the scores associated and sort it using the scores
    pairs = [(i, scores[i]) for i in range(len(distance_matrix))]
    pairs.sort(key=lambda pair: pair[1])

    # Get selected clients params
    selected_clients.append(pairs[0][0])

    # delete selected client distances
    del distances[pairs[0][2]]
    for row in distances:
        del row[pairs[0][2]]

    np.delete(clients_params, pairs[0][2], 0)

    return trim_mean_f(np.array(selected_clients), m / len(selected_clients), axis=0)
