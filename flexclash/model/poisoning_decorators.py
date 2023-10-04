import functools
from copy import deepcopy

from flex.common.utils import check_min_arguments
from flex.data import Dataset
from flex.model import FlexModel


def model_poisoner(func):
    min_args = 1
    assert check_min_arguments(
        func, min_args
    ), f"The decorated function: {func.__name__} is expected to have at least {min_args} argument/s."

    @functools.wraps(func)
    def _poison_FlexModel_(
        client_model: FlexModel,
        client_dataset: Dataset,
        *args,
        **kwargs,
    ):
        poisoned_client_model = func(deepcopy(client_model))
        assert isinstance(
            poisoned_client_model, FlexModel
        ), "The decorated function: {func.__name__} must return a FlexModel object."
        client_model.update(poisoned_client_model)

    return _poison_FlexModel_
