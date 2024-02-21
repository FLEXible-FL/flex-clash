# FLEX-Clash

flex-clash is an implementation of the state-of-the-art adversarial attacks and defences in FL. It is intended to extend the [FLEXible](https://github.com/FLEXible-FL/FLEXible) framework.



## Features
In the following we detail the defences implemented:

|  Defence |  Description  | Citation |
|----------|:-----------------------------------:|------:|
| Median    | It is a robust-aggregation operator based on replacing the arithmetic mean by the median of the model updates, which choose the value that represents the centre of the distribution. | [Byzantine-robust distributed learning: Towards optimal statistical rates.](https://proceedings.mlr.press/v80/yin18a.html) |
| Trimmed mean | It is a version of the arithmetic mean, consisting of filtering a fixed percentage of extreme values both below and above the data distribution. | [Byzantine-robust distributed learning: Towards optimal statistical rates.](https://proceedings.mlr.press/v80/yin18a.html) |
| MultiKrum | It sorts the clients according to the geometric distances of their model updates. Hence, it employs an aggregation parameter, which specifies the number of clients to be aggregated (the first ones after being sorted) resulting in the aggregated model.  | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf) |
| Bulyan | It is a  federated aggregation operator to prevent poisoning attacks, combining the MultiKrum federated aggregation operator and the trimmed-mean. Hence, it sorts the clients according to their geometric distances, and according to a 𝑓 parameter filters out the 2𝑓 clients of the tails of the sorted distribution of clients and aggregates the rest of them.| [The Hidden Vulnerability of Distributed Learning in Byzantium](https://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf) | 


## Installation

In order to install this repo locally:

``
    pip install -e .
``

FLEX-Clash is available on the PyPi repository and can be easily installed using pip:

