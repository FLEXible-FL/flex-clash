# flex-clash

flex-clash is an implementation of the state-of-the-art adversarial attacks and defences in FL. It is intended to extend the [FLEXible](https://github.com/FLEXible-FL/FLEXible) framework.

In order to install this repo locally:

``
    pip install -e ".[develop]"
``

In the following we detail the defences implemented:

|  Defence |    Description   |  Citation |
|----------|:-------------:|------:|
| MultiKrum | It sorts the clients according to the geometric distances of their model updates. Hence, it employs an aggregation parameter, which specifies the number of clients to be aggregated (the first ones after being sorted) resulting in the aggregated model.  | [Machine Learning with Adversaries:
Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf) |

