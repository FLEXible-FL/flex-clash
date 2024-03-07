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
