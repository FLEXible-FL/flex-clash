import functools

from flex.common.utils import check_min_arguments
from flex.data import Dataset


def backdoor_data(func):
    min_args = 2
    assert check_min_arguments(func, min_args), f"The decorated function: {func.__name__} is expected to have at least {min_args} argument/s."

    @functools.wraps(func)
    def _poison_Dataset_(
        client_dataset: Dataset,
        *args,
        **kwargs,
    ):
        features = []
        labels = []
        for feature, label in client_dataset:
            try:
                new_feat, new_label = func(feature, label, *args, **kwargs)
            except ValueError:
                raise ValueError("The decorated function: {func.__name__} must return two values.")
            features.append(new_feat)
            labels.append(new_label)
        return Dataset(X_data=features, y_data=labels)

    return _poison_Dataset_