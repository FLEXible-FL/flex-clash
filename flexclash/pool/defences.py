"""File that contains the adapted server defences based on aggregators in FLEXible.

For the moment, we include:
- Median
- Trimmed mean
- Krum
- Bulyan
"""

from flex.pool.decorators import aggregate_weights

@aggregate_weights
def median(list_of_weights: list):
    import numpy as np

    if len(list_of_weights) == 0:
        return None
    number_of_layers = len(list_of_weights[0])
    layers = [np.array([list_of_weights[i][j] for i in range(len(list_of_weights))]) for j in range(number_of_layers)]
    return [np.median(layer, axis=0) for layer in layers]


@aggregate_weights
def trimmed_mean(list_of_weights: list, trim_proportion=0.1):
    from scipy.stats import trim_mean
    import numpy as np

    if len(list_of_weights) == 0:
        return None
    number_of_layers = len(list_of_weights[0])
    layers = [np.array([list_of_weights[i][j] for i in range(len(list_of_weights))]) for j in range(number_of_layers)]
    return [trim_mean(layer, trim_proportion, axis=0) for layer in layers]

@aggregate_weights
def multikrum(list_of_weights: list, f=1, m=5):
    import numpy as np
    from scipy.stats import trim_mean

    number_of_layers = len(list_of_weights[0])
    clients_params = [np.array([list_of_weights[j][i] for i in range(number_of_layers)])  for j in range(len(list_of_weights))]
    num_clients = len(clients_params)

    #Calculate matrix of distances
    distances = [list() for i in range(num_clients)]
    for i in range(num_clients-1):
        distance = distances[i]
        for j in range(i+1, num_clients):
            dist = 0
            for k in range(number_of_layers):
                dist += np.linalg.norm(clients_params[i][k] - clients_params[j][k])
            distance.append(dist)
            #Simetric matrix
            distances[j].append(dist)


    scores = [list() for i in range(num_clients)]
    num_selected = num_clients - f - 2
    for i in range(num_clients):
        completed_scores = distances[i]
        completed_scores.sort()
        scores[i] = sum(completed_scores[:num_selected])

    #We associate the clients params with the scores associated and sort it using the scores
    pairs = [(clients_params[i], scores[i]) for i in range(num_clients)]
    pairs.sort(key=lambda pair: pair[1])

    #Get selected clients params
    selected_clients = [i[0] for i in pairs[:m]]

    return np.mean(selected_clients, axis = 0)

@aggregate_weights
def bulyan(list_of_weights: list, f=1, m=5):
    import numpy as np
    from scipy.stats import trim_mean

    number_of_layers = len(list_of_weights[0])
    clients_params = [np.array([list_of_weights[j][i] for i in range(number_of_layers)])  for j in range(len(list_of_weights))]
    num_clients = len(clients_params)

    #Calculate matrix of distances
    distances = [list() for i in range(num_clients)]
    for i in range(num_clients-1):
        distance = distances[i]
        for j in range(i+1, num_clients):
            dist = 0
            for k in range(number_of_layers):
                dist += np.linalg.norm(clients_params[i][k] - clients_params[j][k])
            distance.append(dist)
            #Simetric matrix
            distances[j].append(dist)

    #Calculate score matrix
    selected_clients = []
    while len(selected_clients) < num_clients - 2*m:
        scores = [list() for i in range(num_clients)]
        num_selected = num_clients - f - 2
        for i in range(num_clients):
            completed_scores = distances[i]
            completed_scores.sort()
            scores[i] = sum(completed_scores[:num_selected])

    #We associate the clients params with the scores associated and sort it using the scores
    pairs = [(clients_params[i], scores[i]) for i in range(len(distances))]
    pairs.sort(key=lambda pair: pair[1])

    #Get selected clients params
    selected_clients.append(pairs[0][0])

    #delete selected client distances
    del distances[pairs[0][2]]
    for row in distances:
        del row[pairs[0][2]]

    clients_params = np.delete(clients_params, pairs[0][2], 0)

    return trim_mean(np.array(selected_clients), m/len(selected_clients), axis=0)



