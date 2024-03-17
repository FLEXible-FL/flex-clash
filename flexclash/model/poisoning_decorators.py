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

import functools
from copy import deepcopy

from flex.common.utils import check_min_arguments
from flex.data import Dataset
from flex.model import FlexModel


def model_poisoner(func):
    """
    Decorator that applies a poisoning function to a FlexModel object. The poisoning function should take a FlexModel object
    and return a poisoned FlexModel object.

    Note: The FlexModel object is passed by copy, so any direct modification will not affect the original model.

    Args:
    ----
        func: The poisoning function to be applied.

    Returns:
    -------
        A decorated function that applies the poisoning function to a FlexModel object.

    Raises:
    ------
        AssertionError: If the decorated function does not have at least one argument.
        AssertionError: If the decorated function does not return a FlexModel object.
    """

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
