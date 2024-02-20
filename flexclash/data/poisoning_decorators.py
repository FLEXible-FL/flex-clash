import functools

from flex.common.utils import check_min_arguments
from flex.data import Dataset

from flexclash.data.dataset import PoisonedDataset


def data_poisoner(func):
    min_args = 2
    assert check_min_arguments(
        func, min_args
    ), f"The decorated function: {func.__name__} is expected to have at least {min_args} argument/s."

    @functools.wraps(func)
    def _poison_Dataset_(
        client_dataset: Dataset,
        *args,
        **kwargs,
    ):
        def poison_func(label, feature):
            try:
                new_label, new_feature = func(label, feature, *args, **kwargs)
            except ValueError:
                raise ValueError(
                    "The decorated function: {func.__name__} must return two values: features, labels."
                )
            return new_label, new_feature

        poisoned_dataset = PoisonedDataset(
            X_data=client_dataset.X_data,
            y_data=client_dataset.y_data,
            poisoning_function=poison_func,
        )
        return poisoned_dataset

    return _poison_Dataset_
